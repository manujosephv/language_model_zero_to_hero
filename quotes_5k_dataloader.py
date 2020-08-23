import csv
from tqdm.autonotebook import tqdm


class Quote:
    def __init__(self, quote, author, tags, misc):
        self.quote = quote
        self.author = author
        self.tags = tags
        self.misc = misc

    def __repr__(self):
        return (
            f"<Quote object; quote={self.quote}, author={self.author}, tags={self.tags}"
        )

    def __str__(self):
        return f"{self.quote} - {self.author}"


class QuoteDB:
    MOTIVATIONAL = {
        "love",
        "life",
        "inspirational",
        "happiness",
        "hope",
        "inspirational quotes",
        "inspiration",
        "motivational",
        "life lessons",
        "motivation",
        "dreams",
        "life lessons",
    }
    SERIOUS = {
        "philosophy",
        "life",
        "god",
        "time",
        "faith",
        "death",
        "poetry",
        "religion",
        "knowledge",
        "fear",
        "change",
        "spirituality",
        "freedom",
        "purpose",
        "good",
        "peace",
        "money",
        "spiritual",
    }

    FUNNY = {"humor", "romance", "funny"}

    def __init__(self, filepath, preprocessor=None):
        super().__init__()
        self.quotes = {"MOTIVATIONAL": [], "SERIOUS": [], "FUNNY": [], "default": []}
        self.filepath = filepath
        self.preprocessor = preprocessor
        self.read_file()

    def read_file(self):
        self._skipped_lines = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",")
            for row in tqdm(csv_reader):
                try:
                    quote = row[0]
                    author = row[1]
                    tags = row[2]
                    misc = row[3]
                    quote = Quote(
                        quote=quote.lower().strip() if self.preprocessor is None else self.preprocessor(quote),
                        author=author.lower().strip(),
                        tags=tags.lower().strip(),
                        misc=misc.lower().strip(),
                    )
                    self.add(quote)
                except Exception:
                    self._skipped_lines.append(row)
        print(f"Skipped {len(self._skipped_lines)} quotes")

    def add(self, quote):
        if not isinstance(quote, Quote):
            raise ValueError("qute shoud be an instance of <Quote>")
        tags = set([tag.replace("-", " ").strip() for tag in quote.tags.split(",")])
        _any = False
        if len(tags.intersection(self.MOTIVATIONAL)) > 0:
            self.quotes["MOTIVATIONAL"].append(quote)
            _any = True
        if len(tags.intersection(self.SERIOUS)) > 0:
            self.quotes["SERIOUS"].append(quote)
            _any = True
        if len(tags.intersection(self.FUNNY)) > 0:
            self.quotes["FUNNY"].append(quote)
            _any = True
        if not _any:
            self.quotes["default"].append(quote)

    def get_persona_corpus(self, persona):
        if persona not in self.quotes.keys():
            ValueError(f"`persona` needs to be one of {self.quotes.keys()}")
        return [quote.quote for quote in self.quotes[persona]]

