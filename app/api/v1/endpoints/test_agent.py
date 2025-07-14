import json
import os
import shutil
import time
import unittest
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


from app.db_init import init_db
import sqlite3

# Use an in-memory SQLite database for testing
TEST_DB_URL = "file::memory:?cache=shared"

def get_test_db_connection():
    conn = sqlite3.connect(TEST_DB_URL, uri=True)
    conn.row_factory = sqlite3.Row
    return conn

class TestAgentEndpoints(unittest.TestCase):
    def setUp(self):
        # Patch the database connection functions
        self.db_patcher = patch("app.database.get_db_connection", get_test_db_connection)
        self.db_bg_patcher = patch("app.database.get_db_connection_for_bg", get_test_db_connection)
        self.db_patcher.start()
        self.db_bg_patcher.start()

        # Initialize a fresh database schema for each test
        init_db()

        self.kb_path = "knowledge_base_test"
        if not os.path.exists(self.kb_path):
            os.makedirs(self.kb_path)
        self.kb_patcher = patch("app.api.v1.endpoints.agent.kb_path", self.kb_path)
        self.kb_patcher.start()

    def tearDown(self):
        self.kb_patcher.stop()
        self.db_patcher.stop()
        self.db_bg_patcher.stop()

        if os.path.exists(self.kb_path):
            shutil.rmtree(self.kb_path)

        # The in-memory database is discarded automatically when the connection is closed
        # No need to explicitly delete tables

    @patch("app.api.v1.endpoints.agent.get_completion", new_callable=AsyncMock)
    def test_full_task_execution_flow(self, mock_get_completion):
        # --- Mock LLM Responses ---
        # 1. Plan Generation
        mock_plan = [
            {
                "thought": "First, I'll use Python to calculate 2 + 2.",
                "action": "python_code_interpreter",
                "action_input": {"code": "print(2 + 2)"},
            },
            {
                "thought": "Next, I'll write the result into a document.",
                "action": "write_document",
                "action_input": {
                    "title": "Calculation Result",
                    "content": "The result is: {step_1_observation}",
                },
            },
        ]
        # 2. Final Report Generation
        mock_final_report = "# Final Report\n\nThe calculation result is 4."

        # Configure the mock to return different values on subsequent calls
        mock_get_completion.side_effect = [
            json.dumps(mock_plan),
            mock_final_report,
        ]

        # --- Start Task ---
        goal = "Calculate 2+2 and report"
        start_payload = {
            "goal": goal,
            "api_config": {
                "providers": [
                    {
                        "id": "openai-1",
                        "name": "OpenAI",
                        "baseUrl": "https://api.openai.com/v1",
                        "apiKey": "fake_key",
                        "models": ["gpt-4"],
                    }
                ],
                "assignments": {
                    "chat": {"providerId": "openai-1", "modelName": "gpt-4"}
                },
                "keys": {"tavily": "fake_tavily_key"},
            },
            "conversation_id": "test-conversation-123",
        }
        response = client.post("/api/v1/agent/start-task", json=start_payload)
        self.assertEqual(response.status_code, 202)
        task_id = response.json()["task_id"]
        self.assertIsNotNone(task_id)

        # --- Poll for Status ---
        timeout = 30  # seconds
        start_time = time.time()
        final_status = None
        while time.time() - start_time < timeout:
            response = client.get(f"/api/v1/agent/get-task-status/{task_id}")
            self.assertEqual(response.status_code, 200)
            status_data = response.json()
            if status_data["status"] == "completed":
                final_status = status_data
                break
            time.sleep(1)

        # --- Assert Final State ---
        self.assertIsNotNone(final_status, "Task did not complete within timeout")
        self.assertEqual(final_status["status"], "completed")
        self.assertEqual(len(final_status["steps"]), 2)
        self.assertEqual(final_status["steps"][0]["status"], "completed")
        self.assertIn("4", final_status["steps"][0]["observation"])
        self.assertEqual(final_status["steps"][1]["status"], "completed")
        self.assertIn("The result is: 4", final_status["steps"][1]["observation"])

        # --- Verify Final Report File ---
        sanitized_goal = "".join(
            c for c in goal if c.isalnum() or c in (" ", "_")
        ).rstrip()
        filename = f"{sanitized_goal.replace(' ', '_')}_report.md"
        expected_file_path = os.path.join(self.kb_path, filename)

        self.assertTrue(
            os.path.exists(expected_file_path), "Final report file was not created."
        )
        with open(expected_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(
            content, mock_final_report, "Final report content does not match."
        )


if __name__ == "__main__":
    unittest.main()
