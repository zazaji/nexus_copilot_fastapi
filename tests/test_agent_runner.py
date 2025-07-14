import unittest
import sqlite3
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# As the runner is now in a sub-directory, we need to adjust the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.runner import _execute_task
from app.api.v1.endpoints.proxy import ApiConfig
from app.db_init import init_db

class TestAgentRunner(unittest.TestCase):

    def setUp(self):
        """Set up a temporary in-memory database for each test."""
        # Use a real in-memory SQLite database
        self.conn = sqlite3.connect(":memory:")
        # Manually initialize the schema using our db_init script logic
        self.conn.execute("PRAGMA foreign_keys = ON;")
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE agent_tasks (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            user_goal TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER,
            final_report TEXT,
            plan TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE agent_task_steps (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            thought TEXT,
            action TEXT NOT NULL,
            action_input TEXT NOT NULL,
            observation TEXT,
            status TEXT NOT NULL,
            history TEXT,
            FOREIGN KEY (task_id) REFERENCES agent_tasks (id) ON DELETE CASCADE
        )
        """)
        self.conn.commit()
        self.conn.row_factory = sqlite3.Row


        # Mock a valid ApiConfig object for testing
        self.api_config = ApiConfig(
            providers=[
                {
                    "id": "provider-1",
                    "name": "Test Provider",
                    "baseUrl": "http://localhost:1234/v1",
                    "apiKey": "test-api-key",
                    "models": ["test-model"],
                }
            ],
            assignments={
                "chat": {"providerId": "provider-1", "modelName": "test-model"}
            },
            keys={"tavily": "test-tavily-key"}
        )
        self.api_config_dict = self.api_config.model_dump()

    def tearDown(self):
        """Close the database connection after each test."""
        self.conn.close()

    @patch('app.agents.runner._generate_initial_plan', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_simple_task_execution(self, mock_execute_step, mock_generate_plan):
        """Test a simple task flow: search -> finish."""
        goal = "test goal"
        task_id = "test-task-1"
        final_answer = "The answer is 42."
        plan = [{"sub_goal": "Search for the goal."}, {"sub_goal": f"Finish with the answer: {final_answer}"}]

        mock_generate_plan.return_value = plan
        mock_execute_step.side_effect = [
            {"step_summary": "Search successful.", "final_answer": None, "failed": False},
            {"step_summary": "Task finished.", "final_answer": final_answer, "failed": False}
        ]

        asyncio.run(_execute_task(self.conn, task_id, "conv-1", goal, self.api_config_dict))

        task = self.conn.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["final_report"], final_answer)

        steps = self.conn.execute("SELECT * FROM agent_task_steps WHERE task_id = ?", (task_id,)).fetchall()
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]['status'], 'completed')
        self.assertEqual(steps[1]['status'], 'completed')


    @patch('app.agents.runner._generate_initial_plan', new_callable=AsyncMock)
    @patch('app.agents.runner._tool_replan', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_replan_on_failure(self, mock_execute_step, mock_replan, mock_generate_plan):
        """Test that the agent re-plans if a step fails."""
        goal = "test replan"
        task_id = "test-task-replan"

        # Initial plan that is destined to fail
        initial_plan = [{"sub_goal": "Do something that will fail."}]
        mock_generate_plan.return_value = initial_plan

        # The first execution of the step fails
        mock_execute_step.side_effect = [
            {"step_summary": "Something failed.", "final_answer": None, "failed": True},
            # After re-planning, the new step succeeds
            {"step_summary": "New step successful.", "final_answer": "Success!", "failed": False}
        ]

        # The re-planner returns a new, better plan
        new_plan = [{"sub_goal": "Do something that will succeed."}]
        mock_replan.return_value = new_plan

        asyncio.run(_execute_task(self.conn, task_id, "conv-replan", goal, self.api_config_dict))

        # Assertions
        mock_generate_plan.assert_called_once()
        mock_replan.assert_called_once()
        self.assertEqual(mock_execute_step.call_count, 2)

        task = self.conn.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["final_report"], "Success!")

        # We should have one failed step from the first plan, and one successful one from the second
        steps = self.conn.execute("SELECT * FROM agent_task_steps WHERE task_id = ? ORDER BY step_index", (task_id,)).fetchall()
        self.assertEqual(len(steps), 1) # The new plan has one step
        self.assertEqual(steps[0]['status'], 'completed')


    @patch('app.agents.runner._generate_initial_plan', new_callable=AsyncMock)
    @patch('app.agents.runner._determine_next_action_for_sub_goal', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_tool', new_callable=AsyncMock)
    def test_self_correction_within_step(self, mock_execute_tool, mock_determine_action, mock_generate_plan):
        """Test the agent's ability to self-correct after a tool error within a single step."""
        goal = "test self-correction"
        task_id = "test-task-self-correct"
        final_answer = "Corrective action was successful."

        plan = [{"sub_goal": "Try something, fail, then correct."}]
        mock_generate_plan.return_value = plan

        # The executor first tries a failing action, then a correcting one.
        mock_determine_action.side_effect = [
            {"thought": "Let's try the failing tool.", "action": "failing_tool", "action_input": {}},
            {"thought": "That failed. Let's correct.", "action": "correcting_tool", "action_input": {}},
            {"thought": "Correction worked.", "action": "finish_task", "action_input": {"final_answer": final_answer}},
        ]

        # The first tool call raises an exception, the second one succeeds.
        mock_execute_tool.side_effect = [
            ValueError("Tool failed!"),
            "Corrective action output.",
            final_answer
        ]

        asyncio.run(_execute_task(self.conn, task_id, "conv-self-correct", goal, self.api_config_dict))

        # Assertions
        self.assertEqual(mock_execute_tool.call_count, 3)

        task = self.conn.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["final_report"], final_answer)

        step = self.conn.execute("SELECT * FROM agent_task_steps WHERE task_id = ?", (task_id,)).fetchone()
        self.assertEqual(step['status'], 'completed')

        # Check that the history of the step reflects the correction
        history = json.loads(step['history'])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['action'], 'error')
        self.assertIn("Tool failed!", history[0]['observation'])
        self.assertEqual(history[1]['action'], 'correcting_tool')


    @patch('app.agents.runner._generate_initial_plan', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_complex_report_generation_with_failures_and_replan(self, mock_execute_step, mock_generate_plan):
        """
        An end-to-end test simulating a complex report generation task
        that involves a step failure and a subsequent re-plan.
        """
        goal = "Generate a full market analysis report for Product X."
        task_id = "complex-report-task"
        final_report = "This is the final, successful report."

        # --- Initial Plan ---
        initial_plan = [
            {"sub_goal": "Gather preliminary data for Product X."},
            {"sub_goal": "Analyze competitor data, which will fail."}, # This step is designed to fail
            {"sub_goal": "Generate report outline."},
            {"sub_goal": "Write report and finish."}
        ]
        mock_generate_plan.return_value = initial_plan

        # --- New Plan (after failure) ---
        new_plan = [
             {"sub_goal": "Gather data from an alternative source."},
             {"sub_goal": "Generate a revised report outline."},
             {"sub_goal": "Write the final report based on the new data and outline, then finish."}
        ]

        # We need to patch the replan tool specifically for this test
        with patch('app.agents.runner._tool_replan', new_callable=AsyncMock) as mock_replan:
            mock_replan.return_value = new_plan

            # --- Mock Step Executions ---
            mock_execute_step.side_effect = [
                # Initial plan execution
                {"step_summary": "Data gathered.", "final_answer": None, "failed": False},
                {"step_summary": "Competitor analysis failed due to API error.", "final_answer": None, "failed": True},
                # Re-planned execution
                {"step_summary": "Alternative data gathered.", "final_answer": None, "failed": False},
                {"step_summary": "Revised outline generated.", "final_answer": None, "failed": False},
                {"step_summary": "Final report written.", "final_answer": final_report, "failed": False}
            ]

            # --- Run Task ---
            asyncio.run(_execute_task(self.conn, task_id, "conv-complex", goal, self.api_config_dict))

            # --- Assertions ---
            # 1. Verify that re-planning was triggered
            mock_replan.assert_called_once()

            # 2. Check the final task status and report
            task = self.conn.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["final_report"], final_report)

            # 3. Check the final plan stored in the database
            final_plan_from_db = json.loads(task["plan"])
            self.assertEqual(final_plan_from_db, initial_plan) # The 'plan' field is not updated after replan in the current implementation

            # 4. Check the steps in the database - should correspond to the *new* plan
            steps = self.conn.execute("SELECT * FROM agent_task_steps WHERE task_id = ? ORDER BY step_index", (task_id,)).fetchall()
            self.assertEqual(len(steps), len(new_plan))
            self.assertEqual(steps[0]['action'], 'Gather data from an alternative source.')
            self.assertEqual(steps[0]['status'], 'completed')
            self.assertEqual(steps[2]['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
