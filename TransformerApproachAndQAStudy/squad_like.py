
import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This generic data loading will load a dataset that has the SQuAD.v2 format. 
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
    "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
}


class SquadLikeConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(SquadLikeConfig, self).__init__(**kwargs)


class SquadLike(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        SquadLikeConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            self.config.data_dir = ''
            
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepaths": self.config.data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepaths": self.config.data_files["validation"]}),
        ]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form."""
        key = 0
        for filepath in filepaths:
            logger.info("generating examples from = %s", filepath)
            with open(filepath, encoding="utf-8") as f:
                squad = json.load(f)
                for article in squad["data"]:
                    title = article.get("title", "")
                    for paragraph in article["paragraphs"]:
                        # do not strip leading blank spaces GH-2585
                        context = paragraph["context"]
                        for qa in paragraph["qas"]:
                            answer_starts = [answer["answer_start"]
                                            for answer in qa["answers"]]
                            answers = [answer["text"] for answer in qa["answers"]]
                            # Features currently used are "context", "question", and "answers".
                            # Others are extracted here for the ease of future expansions.
                            yield key, {
                                "title": title,
                                "context": context,
                                "question": qa["question"],
                                "id": qa["id"],
                                "answers": {
                                    "answer_start": answer_starts,
                                    "text": answers,
                                },
                            }
                            key += 1
