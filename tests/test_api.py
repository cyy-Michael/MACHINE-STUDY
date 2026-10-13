# -*- coding: utf-8 -*-
"""接口层测试（需要本地数据集）。

覆盖老师要求的 6 个接口、统一错误响应、CORS 与 WebSocket 进度推送。
"""
from __future__ import annotations

import time
import unittest

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.services import store
from tests.support import requires_data


def wait_for(client: TestClient, experiment_id: str, timeout: float = 60.0) -> dict:
    """轮询接口五直到任务结束。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/experiments/{experiment_id}").json()
        if payload["status"] in ("finished", "failed"):
            return payload
        time.sleep(0.05)
    raise AssertionError(f"实验 {experiment_id} 在 {timeout}s 内没有结束")


@requires_data
class DatasetEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_list_datasets(self):
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload), 7)
        self.assertIn("name", payload[0])
        self.assertIn("n_samples", payload[0])

    def test_list_datasets_filtered_by_task(self):
        response = self.client.get("/api/datasets", params={"task_type": "regression"})
        self.assertEqual(response.status_code, 200)
        names = [item["name"] for item in response.json()]
        self.assertIn("bike_day", names)
        self.assertNotIn("iris", names)

    def test_dataset_detail(self):
        response = self.client.get("/api/datasets/iris")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "iris")
        self.assertEqual(payload["n_samples"], 150)
        self.assertEqual(payload["target"], "class")
        self.assertEqual(len(payload["fields"]), 5)
        self.assertIn("label_distribution", payload)
        self.assertEqual(sum(payload["label_distribution"].values()), 150)

    def test_dataset_detail_reports_missing_counts(self):
        payload = self.client.get("/api/datasets/ckd").json()
        self.assertGreater(payload["missing_total"], 0)
        by_name = {field["name"]: field for field in payload["fields"]}
        self.assertGreater(by_name["rbc"]["missing"], 0)

    def test_unknown_dataset_returns_json_404(self):
        response = self.client.get("/api/datasets/no_such_dataset")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["type"], "DatasetNotFound")


@requires_data
class AlgorithmEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_list_algorithms(self):
        response = self.client.get("/api/algorithms")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 10)

    def test_algorithm_prefix_is_under_api(self):
        """P0 缺陷 #2：路由必须挂在 /api 下。"""
        self.assertEqual(self.client.get("/api/algorithms").status_code, 200)
        self.assertEqual(self.client.get("/algorithms").status_code, 404)

    def test_task_type_reported_in_docs_example(self):
        payload = self.client.get(
            "/api/algorithms", params={"task_type": "classification"}
        ).json()
        self.assertEqual(len(payload), 7)
        self.assertTrue(all("classification" in a["task_type"] for a in payload))

    def test_unknown_task_type_returns_empty_list(self):
        response = self.client.get("/api/algorithms", params={"task_type": "bogus"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])


@requires_data
class ExperimentEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self._created: list[str] = []

    def tearDown(self):
        for experiment_id in self._created:
            store.delete_experiment(experiment_id)

    def _submit(self, **body) -> str:
        response = self.client.post("/api/experiments", json=body)
        self.assertEqual(response.status_code, 202, response.text)
        experiment_id = response.json()["experiment_id"]
        self._created.append(experiment_id)
        return experiment_id

    def test_create_and_poll_experiment(self):
        experiment_id = self._submit(dataset="iris", algorithm="knn", parameters={"k": 5})
        payload = wait_for(self.client, experiment_id)
        self.assertEqual(payload["status"], "finished")
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["result"]["dataset"], "iris")
        self.assertGreater(payload["result"]["metrics"]["accuracy"], 0.8)

    def test_create_returns_poll_and_websocket_urls(self):
        response = self.client.post(
            "/api/experiments", json={"dataset": "iris", "algorithm": "knn"}
        )
        body = response.json()
        self._created.append(body["experiment_id"])
        self.assertIn(body["experiment_id"], body["poll_url"])
        self.assertTrue(body["websocket_url"].startswith("/ws/experiments/"))

    def test_result_is_persisted_to_disk(self):
        experiment_id = self._submit(dataset="iris", algorithm="decision_tree")
        wait_for(self.client, experiment_id)
        self.assertTrue(store.experiment_exists(experiment_id))
        saved = store.load_experiment(experiment_id)
        self.assertEqual(saved["experiment_id"], experiment_id)

    def test_history_endpoint_lists_saved_experiments(self):
        experiment_id = self._submit(dataset="iris", algorithm="knn")
        wait_for(self.client, experiment_id)
        payload = self.client.get("/api/experiments", params={"limit": 100}).json()
        self.assertIn(experiment_id, [item["experiment_id"] for item in payload["experiments"]])

    def test_failed_experiment_reports_error(self):
        """参数非法时后台任务应记录失败而不是让服务崩掉。"""
        experiment_id = self._submit(
            dataset="iris", algorithm="knn", parameters={"k": 0}
        )
        payload = wait_for(self.client, experiment_id)
        self.assertEqual(payload["status"], "failed")
        self.assertIsNotNone(payload["error"])

    def test_websocket_streams_progress_until_finished(self):
        experiment_id = self._submit(dataset="iris", algorithm="knn", parameters={"k": 5})
        types: list[str] = []
        statuses: list[str] = []
        with self.client.websocket_connect(f"/ws/experiments/{experiment_id}") as ws:
            while True:
                message = ws.receive_json()
                types.append(message["type"])
                statuses.append(message.get("status"))
                if message.get("status") in ("finished", "failed"):
                    break
        self.assertIn("finished", statuses)
        self.assertIn("snapshot", types)

    def test_websocket_after_completion_replays_result(self):
        experiment_id = self._submit(dataset="iris", algorithm="knn", parameters={"k": 3})
        wait_for(self.client, experiment_id)
        with self.client.websocket_connect(f"/ws/experiments/{experiment_id}") as ws:
            first = ws.receive_json()
            second = ws.receive_json()
        self.assertEqual(first["type"], "snapshot")
        self.assertEqual(first["status"], "finished")
        self.assertEqual(second["type"], "result")

    def test_websocket_unknown_experiment(self):
        with self.client.websocket_connect("/ws/experiments/nope") as ws:
            message = ws.receive_json()
        self.assertEqual(message["type"], "error")


@requires_data
class ErrorHandlingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_unknown_dataset_on_create(self):
        response = self.client.post(
            "/api/experiments", json={"dataset": "nope", "algorithm": "knn"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["type"], "DatasetNotFound")

    def test_unknown_algorithm_on_create(self):
        response = self.client.post(
            "/api/experiments", json={"dataset": "iris", "algorithm": "nope"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_task_mismatch_on_create(self):
        response = self.client.post(
            "/api/experiments",
            json={"dataset": "iris", "algorithm": "linear_regression"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "ExperimentError")

    def test_invalid_test_size_is_a_validation_error(self):
        response = self.client.post(
            "/api/experiments",
            json={"dataset": "iris", "algorithm": "knn", "test_size": 5},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "ValidationError")

    def test_missing_required_field(self):
        response = self.client.post("/api/experiments", json={"dataset": "iris"})
        self.assertEqual(response.status_code, 400)

    def test_unknown_experiment_returns_json_404(self):
        response = self.client.get("/api/experiments/nope")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["type"], "ExperimentNotFound")

    def test_health(self):
        payload = self.client.get("/health").json()
        self.assertEqual(payload["status"], "ok")
        self.assertGreaterEqual(payload["datasets"], 7)


@requires_data
class CorsTest(unittest.TestCase):
    """前端在 5173 端口开发，必须能跨域访问。"""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_preflight_allows_vue_dev_server(self):
        response = self.client.options(
            "/api/datasets",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertIn(response.status_code, (200, 204))
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    def test_actual_request_carries_cors_header(self):
        response = self.client.get(
            "/api/datasets", headers={"Origin": "http://localhost:5173"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )


if __name__ == "__main__":
    unittest.main()
