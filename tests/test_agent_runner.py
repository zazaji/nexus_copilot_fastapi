import unittest
import sqlite3
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# As the runner is now in a sub-directory, we need to adjust the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.runner import run_task_background, _execute_task
from app.api.v1.endpoints.proxy import ApiConfig

class TestAgentRunner(unittest.TestCase):

    def setUp(self):
        """Set up a temporary in-memory database for each test."""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # Create tables
        self.cursor.execute("""
        CREATE TABLE agent_tasks (
            id TEXT PRIMARY KEY, conversation_id TEXT, user_goal TEXT, status TEXT,
            created_at INTEGER, updated_at INTEGER, final_report TEXT
        )
        """)
        self.cursor.execute("""
        CREATE TABLE agent_task_steps (
            id TEXT PRIMARY KEY, task_id TEXT, step_index INTEGER, thought TEXT,
            action TEXT, action_input TEXT, observation TEXT, status TEXT,
            FOREIGN KEY (task_id) REFERENCES agent_tasks (id)
        )
        """)
        self.conn.commit()

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

    @patch('app.agents.runner.get_completion', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_simple_task_execution(self, mock_execute_step, mock_get_completion):
        """Test a simple task flow: search -> finish."""

        # --- Mock Setup ---
        # 1. First LLM call decides to search
        mock_get_completion.side_effect = [
            json.dumps({
                "thought": "I need to search for the goal.",
                "action": "internet_search",
                "action_input": {"query": "test goal"}
            }),
            # 2. Second LLM call decides to finish
            json.dumps({
                "thought": "I have the search results, I can now finish.",
                "action": "FINISH",
                "action_input": {"final_answer": "The answer is 42."}
            })
        ]

        # Mock the execution of the search tool
        mock_execute_step.return_value = "Search successful: Found relevant information."

        # --- Run the task ---
        task_id = "test-task-1"
        goal = "test goal"

        # We need to run the async function _execute_task directly
        asyncio.run(_execute_task(self.conn, task_id, "conv-1", goal, self.api_config_dict))

        # --- Assertions ---
        # 1. Check final task status
        task = self.cursor.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["final_report"], "The answer is 42.")

        # 2. Check steps recorded in DB
        steps = self.cursor.execute("SELECT * FROM agent_task_steps WHERE task_id = ? ORDER BY step_index", (task_id,)).fetchall()
        self.assertEqual(len(steps), 1)

        step1 = steps[0]
        self.assertEqual(step1["step_index"], 1)
        self.assertEqual(step1["action"], "internet_search")
        self.assertEqual(step1["observation"], "Search successful: Found relevant information.")
        self.assertEqual(step1["status"], "completed")

        # 3. Verify mocks were called correctly
        self.assertEqual(mock_get_completion.call_count, 2)
        mock_execute_step.assert_called_once_with("internet_search", {"query": "test goal"}, self.api_config)


    @patch('app.agents.runner.get_completion', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_tool_error_handling(self, mock_execute_step, mock_get_completion):
        """Test that the agent can handle a tool execution error."""

        # --- Mock Setup ---
        # 1. First LLM call decides to use a tool
        mock_get_completion.side_effect = [
            json.dumps({
                "thought": "I will try to run some python code.",
                "action": "python_code_interpreter",
                "action_input": {"code": "print(1/0)"} # This will cause an error
            }),
            # 2. Second LLM call sees the error and decides to finish
            json.dumps({
                "thought": "The python code failed to execute. I cannot proceed and will report the failure.",
                "action": "FINISH",
                "action_input": {"final_answer": "Failed to execute Python code with error: Division by zero."}
            })
        ]

        # Mock the execution of the tool to return an error message
        error_message = "Error executing tool 'python_code_interpreter': division by zero. Please check your action_input."
        mock_execute_step.return_value = error_message

        # --- Run the task ---
        task_id = "test-task-2"
        goal = "run bad python code"

        asyncio.run(_execute_task(self.conn, task_id, "conv-2", goal, self.api_config_dict))

        # --- Assertions ---
        # 1. Check final task status
        task = self.cursor.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertIn("Failed to execute Python code", task["final_report"])

        # 2. Check the step that failed
        steps = self.cursor.execute("SELECT * FROM agent_task_steps WHERE task_id = ?", (task_id,)).fetchall()
        self.assertEqual(len(steps), 1)

        failed_step = steps[0]
        self.assertEqual(failed_step["action"], "python_code_interpreter")
        self.assertEqual(failed_step["observation"], error_message)
        self.assertEqual(failed_step["status"], "completed") # The step itself completes, but the observation contains the error

        # 3. Verify mocks
        self.assertEqual(mock_get_completion.call_count, 2)
        mock_execute_step.assert_called_once()


    @patch('app.agents.runner.get_completion', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_multi_step_report_generation(self, mock_execute_step, mock_get_completion):
        """Test a more complex flow involving multi-step report generation."""

        # --- Mock Setup ---
        outline = ["Chapter 1: Introduction", "Chapter 2: Conclusion"]
        chapter1_content = "This is the introduction."
        chapter2_content = "This is the conclusion."
        final_report_content = f"# Report\n\n## Chapter 1: Introduction\n{chapter1_content}\n\n## Chapter 2: Conclusion\n{chapter2_content}"

        mock_get_completion.side_effect = [
            # 1. Decide to generate an outline
            json.dumps({
                "thought": "I need to create a report. I will start with an outline.",
                "action": "generate_report_outline",
                "action_input": {"goal": "write a report"}
            }),
            # 2. Decide to write chapter 1
            json.dumps({
                "thought": "Outline is ready. I will write the first chapter.",
                "action": "write_report_chapter",
                "action_input": {"goal": "write a report", "outline": outline, "chapter_title": "Chapter 1: Introduction"}
            }),
            # 3. Decide to write chapter 2
            json.dumps({
                "thought": "First chapter is done. Now for the second chapter.",
                "action": "write_report_chapter",
                "action_input": {"goal": "write a report", "outline": outline, "chapter_title": "Chapter 2: Conclusion", "previous_chapters": chapter1_content}
            }),
            # 4. Decide to save the final report
            json.dumps({
                "thought": "All chapters are written. I will now save the consolidated report.",
                "action": "save_to_knowledge_base",
                "action_input": {"filename": "final_report.md", "content": final_report_content}
            }),
            # 5. Finish the task
            json.dumps({
                "thought": "The report has been saved.",
                "action": "FINISH",
                "action_input": {"final_answer": "Successfully generated and saved the report."}
            })
        ]

        mock_execute_step.side_effect = [
            json.dumps(outline),
            chapter1_content,
            chapter2_content,
            "Success: File 'final_report.md' saved.",
        ]

        # --- Run the task ---
        task_id = "test-task-3"
        goal = "write a report"
        asyncio.run(_execute_task(self.conn, task_id, "conv-3", goal, self.api_config_dict))

        # --- Assertions ---
        task = self.cursor.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["final_report"], "Successfully generated and saved the report.")

        steps = self.cursor.execute("SELECT * FROM agent_task_steps WHERE task_id = ? ORDER BY step_index", (task_id,)).fetchall()
        self.assertEqual(len(steps), 4)

        self.assertEqual(steps[0]["action"], "generate_report_outline")
        self.assertEqual(steps[1]["action"], "write_report_chapter")
        self.assertEqual(steps[2]["action"], "write_report_chapter")
        self.assertEqual(steps[3]["action"], "save_to_knowledge_base")

        self.assertEqual(mock_get_completion.call_count, 5)
        self.assertEqual(mock_execute_step.call_count, 4)


if __name__ == '__main__':
    unittest.main()
