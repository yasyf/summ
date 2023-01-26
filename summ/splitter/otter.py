from summ.splitter.splitter import Splitter


class OtterSplitter(Splitter):
    """Adds Splitter support for transcripts exported from otter.ai.

    To filter out your own remarks, pass a list of speakers to exclude.
    """

    def __init__(self, speakers_to_exclude: list[str] = []) -> None:
        super().__init__()
        self.exclude = speakers_to_exclude

    def get_chunks(self, title: str, text: str) -> list[str]:
        return [
            speaker_chunk[1]
            for utterance in text.split("\n\n")
            for speaker_chunk in [utterance.split("\n")]
            if "\n" in utterance
            and not any(sp in speaker_chunk[0].lower() for sp in self.exclude)
        ]
