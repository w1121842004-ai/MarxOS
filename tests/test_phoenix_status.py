import os
import unittest
from unittest.mock import patch

import marxos_phoenix as phoenix


class PhoenixStatusTests(unittest.TestCase):
    def test_startup_status_reports_disabled_when_flag_is_off(self):
        with patch.dict(os.environ, {}, clear=False):
            self.assertEqual(phoenix.startup_status_lines(), ["Phoenix tracing: disabled"])

    def test_startup_status_reports_reachable_ui_when_enabled(self):
        with (
            patch.dict(
                os.environ,
                {
                    "MARXOS_PHOENIX_ENABLED": "1",
                    "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:6006/v1/traces",
                    "MARXOS_PHOENIX_PROJECT_NAME": "MarxOS",
                },
                clear=False,
            ),
            patch.object(phoenix.trace_manager, "init_error", return_value=""),
            patch("marxos_phoenix.phoenix_ui_reachable", return_value=True),
        ):
            lines = phoenix.startup_status_lines()

        self.assertIn("Phoenix tracing: enabled (project=MarxOS)", lines)
        self.assertIn("Phoenix collector: http://127.0.0.1:6006/v1/traces", lines)
        self.assertIn("Phoenix UI reachable: http://127.0.0.1:6006", lines)

    def test_startup_status_warns_when_ui_is_unreachable(self):
        with (
            patch.dict(
                os.environ,
                {
                    "MARXOS_PHOENIX_ENABLED": "1",
                    "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:6006/v1/traces",
                },
                clear=False,
            ),
            patch.object(phoenix.trace_manager, "init_error", return_value=""),
            patch("marxos_phoenix.phoenix_ui_reachable", return_value=False),
        ):
            lines = phoenix.startup_status_lines()

        self.assertIn("Phoenix warning: UI not reachable at http://127.0.0.1:6006", lines)


if __name__ == "__main__":
    unittest.main()
