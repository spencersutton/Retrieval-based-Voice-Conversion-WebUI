import json
import locale
import os


def load_language_list(language: str) -> dict[str, str]:
    with open(f"./i18n/locale/{language}.json", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language: None = None) -> None:
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        if not os.path.exists(f"./i18n/locale/{language}.json"):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key: str) -> str:
        return self.language_map.get(key, key)

    def __repr__(self) -> str:
        return f"Use Language: {self.language}"
