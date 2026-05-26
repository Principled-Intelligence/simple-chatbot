import unittest

from simple_chatbot.responses import normalize_input, InvalidInputError


class NormalizeInputTests(unittest.TestCase):
    def test_string_input_becomes_user_message(self):
        self.assertEqual(
            normalize_input("hello"),
            [{"role": "user", "content": "hello"}],
        )

    def test_list_of_strings_becomes_user_messages(self):
        self.assertEqual(
            normalize_input(["hi", "again"]),
            [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
        )

    def test_message_item_with_role_user(self):
        self.assertEqual(
            normalize_input([{"type": "message", "role": "user", "content": "hi"}]),
            [{"role": "user", "content": "hi"}],
        )

    def test_message_item_with_role_assistant(self):
        self.assertEqual(
            normalize_input([{"type": "message", "role": "assistant", "content": "ok"}]),
            [{"role": "assistant", "content": "ok"}],
        )

    def test_message_item_with_content_parts(self):
        # Content parts passed through unchanged
        parts = [{"type": "input_text", "text": "hello"}]
        self.assertEqual(
            normalize_input([{"type": "message", "role": "user", "content": parts}]),
            [{"role": "user", "content": parts}],
        )

    def test_mixed_list(self):
        self.assertEqual(
            normalize_input([
                "first",
                {"type": "message", "role": "assistant", "content": "second"},
                {"type": "message", "role": "user", "content": "third"},
            ]),
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
        )

    def test_unknown_item_type_raises(self):
        with self.assertRaises(InvalidInputError):
            normalize_input([{"type": "made_up_type", "blob": 1}])

    def test_non_string_non_dict_item_raises(self):
        with self.assertRaises(InvalidInputError):
            normalize_input([42])


if __name__ == "__main__":
    unittest.main()
