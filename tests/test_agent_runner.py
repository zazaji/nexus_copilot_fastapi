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

    @patch('app.agents.runner._generate_initial_plan', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_simple_task_execution(self, mock_execute_step, mock_generate_plan):
        """Test a simple task flow: search -> finish."""

        # --- Mock Setup ---
        goal = "test goal"
        task_id = "test-task-1"
        final_answer = "The answer is 42."
        plan = [
            {"sub_goal": "Search for the goal."},
            {"sub_goal": "Finish the task."}
        ]
        mock_generate_plan.return_value = plan

        mock_execute_step.side_effect = [
            {"step_summary": "Search successful: Found relevant information.", "final_answer": None},
            {"step_summary": "Task finished.", "final_answer": final_answer}
        ]

        # --- Run the task ---
        asyncio.run(_execute_task(self.conn, task_id, "conv-1", goal, self.api_config_dict))

        # --- Assertions ---
        # 1. Check final task status
        task = self.cursor.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["final_report"], final_answer)

        # 2. Check steps recorded in DB
        steps = self.cursor.execute("SELECT * FROM agent_task_steps WHERE task_id = ? ORDER BY step_index", (task_id,)).fetchall()
        self.assertEqual(len(steps), 2)


    @patch('app.agents.runner._generate_initial_plan', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_tool_error_handling(self, mock_execute_step, mock_generate_plan):
        """Test that the agent can handle a tool execution error."""

        # --- Mock Setup ---
        goal = "run bad python code"
        task_id = "test-task-2"
        final_answer = "Failed to execute Python code with error: division by zero."
        plan = [
            {"sub_goal": "Run some python code that will fail."},
            {"sub_goal": "Finish the task."}
        ]
        mock_generate_plan.return_value = plan

        mock_execute_step.side_effect = [
            {"step_summary": "Error executing tool 'python_code_interpreter': division by zero.", "final_answer": None},
            {"step_summary": "Task finished.", "final_answer": final_answer}
        ]

        # --- Run the task ---
        asyncio.run(_execute_task(self.conn, task_id, "conv-2", goal, self.api_config_dict))

        # --- Assertions ---
        # 1. Check final task status
        task = self.cursor.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertIn("Failed to execute Python code", task["final_report"])

        # 2. Check the step that failed
        steps = self.cursor.execute("SELECT * FROM agent_task_steps WHERE task_id = ?", (task_id,)).fetchall()
        self.assertEqual(len(steps), 2)


    @patch('app.agents.runner._generate_initial_plan', new_callable=AsyncMock)
    @patch('app.agents.runner._execute_step', new_callable=AsyncMock)
    def test_multi_step_report_generation(self, mock_execute_step, mock_generate_plan):
        """Test a more complex flow involving multi-step report generation."""

        # --- Mock Setup ---
        goal = "write a report"
        task_id = "test-task-3"
        outline = ["Chapter 1: Introduction", "Chapter 2: Conclusion"]
        chapter1_content = "This is the introduction."
        chapter2_content = "This is the conclusion."
        final_report_content = f"# Report\n\n## Chapter 1: Introduction\n{chapter1_content}\n\n## Chapter 2: Conclusion\n{chapter2_content}"
        final_answer = "Successfully generated and saved the report."
        plan = [
            {"sub_goal": "Generate an outline for the report."},
            {"sub_goal": "Write the first chapter."},
            {"sub_goal": "Write the second chapter."},
            {"sub_goal": "Save the final report."},
            {"sub_goal": "Finish the task."}
        ]
        mock_generate_plan.return_value = plan

        mock_execute_step.side_effect = [
            {"step_summary": f"Sub-goal: Generate an outline for the report.\nOutcome: {json.dumps(outline)}", "final_answer": None},
            {"step_summary": f"Sub-goal: Write the first chapter.\nOutcome: {chapter1_content}", "final_answer": None},
            {"step_summary": f"Sub-goal: Write the second chapter.\nOutcome: {chapter2_content}", "final_answer": None},
            {"step_summary": "Sub-goal: Save the final report.\nOutcome: Success: File 'final_report.md' saved.", "final_answer": None},
            {"step_summary": "Task finished.", "final_answer": final_answer}
        ]

        # --- Run the task ---
        asyncio.run(_execute_task(self.conn, task_id, "conv-3", goal, self.api_config_dict))

        # --- Assertions ---
        task = self.cursor.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["final_report"], final_answer)

        steps = self.cursor.execute("SELECT * FROM agent_task_steps WHERE task_id = ? ORDER BY step_index", (task_id,)).fetchall()
        self.assertEqual(len(steps), 5)


    def test_complex_investment_report(self):
        """
        Test a complex task of generating an investment report,
        covering the full planner -> executor -> tool -> report flow.
        """
        goal = "请查找中国股市酒类相关股票的股价和其他相关研报，然后给出详细的投资分析报告。"
        task_id = "investment-report-task"

        # --- Mock Data ---
        search_results = "茅台股价：2000元，五粮液股价：300元。研报摘要：酒类市场前景广阔..."
        analyzed_data = "分析结果：茅台和五粮液是龙头企业，具有长期投资价值。"
        outline = ["1. 行业概览", "2. 重点公司分析", "3. 投资建议"]
        chapter1 = "第一章内容..."
        chapter2 = "第二章内容..."
        chapter3 = "第三章内容..."
        final_report = f"# 投资分析报告\n\n## 1. 行业概览\n{chapter1}\n\n## 2. 重点公司分析\n{chapter2}\n\n## 3. 投资建议\n{chapter3}"

        # --- Mock LLM Calls ---
        # 1. Planner's response (the plan)
        planner_response = {
            "plan": [
                {"sub_goal": "使用互联网搜索查找中国股市酒类相关股票的最新股价。"},
                {"sub_goal": "使用互联网搜索查找相关的行业研究报告。"},
                {"sub_goal": "综合搜索结果，分析市场趋势和重点公司。"},
                {"sub_goal": "为投资报告生成一个大纲，包括行业概览、重点公司分析和投资建议。"},
                {"sub_goal": "撰写'行业概览'章节。"},
                {"sub_goal": "撰写'重点公司分析'章节。"},
                {"sub_goal": "撰写'投资建议'章节。"},
                {"sub_goal": f"将所有章节合并成最终报告，并调用 finish_task 工具。"},
            ]
        }

        # Executor's responses for each sub-goal
        executor_responses = [
            # Step 1: Search stock prices
            {"thought": "...", "action": "internet_search", "action_input": {"query": "中国股市酒类股票股价"}},
            {"thought": "...", "action": "COMPLETE_SUB_GOAL", "action_input": {}},
            # Step 2: Search research papers
            {"thought": "...", "action": "internet_search", "action_input": {"query": "中国酒类股票行业研究报告"}},
            {"thought": "...", "action": "COMPLETE_SUB_GOAL", "action_input": {}},
            # Step 3: Analyze data
            {"thought": "...", "action": "python_code_interpreter", "action_input": {"code": "print('分析结果...')"}},
            {"thought": "...", "action": "COMPLETE_SUB_GOAL", "action_input": {}},
            # Step 4: Generate outline
            {"thought": "...", "action": "generate_report_outline", "action_input": {"goal": goal}},
            {"thought": "...", "action": "COMPLETE_SUB_GOAL", "action_input": {}},
            # Step 5: Write chapter 1
            {"thought": "...", "action": "write_report_chapter", "action_input": {"chapter_title": "1. 行业概览"}},
            {"thought": "...", "action": "COMPLETE_SUB_GOAL", "action_input": {}},
            # Step 6: Write chapter 2
            {"thought": "...", "action": "write_report_chapter", "action_input": {"chapter_title": "2. 重点公司分析"}},
            {"thought": "...", "action": "COMPLETE_SUB_GOAL", "action_input": {}},
            # Step 7: Write chapter 3
            {"thought": "...", "action": "write_report_chapter", "action_input": {"chapter_title": "3. 投资建议"}},
            {"thought": "...", "action": "COMPLETE_SUB_GOAL", "action_input": {}},
             # Step 8: Finish
            {"thought": "...", "action": "finish_task", "action_input": {"final_answer": final_report}},
        ]


        # --- Patching ---
        with patch('app.agents.runner._generate_initial_plan', new_callable=AsyncMock) as mock_generate_plan, \
             patch('app.agents.runner._determine_next_action_for_sub_goal', new_callable=AsyncMock) as mock_determine_action, \
             patch('app.agents.runner._execute_tool', new_callable=AsyncMock) as mock_execute_tool:

            # Setup mock return values
            mock_generate_plan.return_value = planner_response["plan"]
            mock_determine_action.side_effect = executor_responses
            mock_execute_tool.side_effect = [
                search_results, # For step 1
                search_results, # For step 2
                analyzed_data,  # For step 3
                json.dumps(outline), # For step 4
                chapter1, # For step 5
                chapter2, # For step 6
                chapter3, # For step 7
                final_report # for step 8
            ]

            # --- Run Task ---
            asyncio.run(_execute_task(self.conn, task_id, "conv-complex", goal, self.api_config_dict))

            # --- Assertions ---
            # 1. Final task status
            task = self.cursor.execute("SELECT * FROM agent_tasks WHERE id = ?", (task_id,)).fetchone()
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["final_report"], final_report)

            # 2. Check number of steps in DB
            steps = self.cursor.execute("SELECT * FROM agent_task_steps WHERE task_id = ?", (task_id,)).fetchall()
            self.assertEqual(len(steps), len(planner_response["plan"]))

            # 3. Verify calls
            mock_generate_plan.assert_called_once_with(goal, self.api_config)
            self.assertEqual(mock_determine_action.call_count, len(executor_responses))
            self.assertEqual(mock_execute_tool.call_count, 8) # 8 tools were called


if __name__ == '__main__':
    unittest.main()
