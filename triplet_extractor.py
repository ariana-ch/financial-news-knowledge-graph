import json
import os
import re

import pandas as pd
from pathlib import Path
from google import genai
from google.genai.types import Schema, Type
from google.genai import types
from typing import List, Optional, Any
import datetime
from downloaders import BaseDataHandler
from preprocessors import ArticlePreprocessor
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get('GEMINI_API_KEY')

# Define the controlled vocabulary from the prompt
entity_types = [
    "COMPANY",
    "PERSON",
    "PRODUCT",
    "FINANCIAL_INSTRUMENT",
    "SECTOR",
    "EVENT",
    "LOCATION",
    "EXCHANGE",
    "ROLE"
]

# Define the controlled vocabulary for event subtypes
event_subtypes = [
    "MergerAcquisition", "ExecutiveChange", "ProductLaunch", "EarningsAnnouncement",
    "Litigation", "SupplyChainDisruption", "Bankruptcy", "CorporateScandal",
    "RegulatoryAction", "TradeSanctions", "MonetaryPolicy", "FiscalPolicy",
    "LegalChange", "War", "CivilUnrest", "PriceDrop", "PriceSurge",
    "VolatilitySpike", "Pandemic", "NaturalDisaster", "CyberAttack",
    "MarketCrash", "RecordHigh", "RecordLow", "CurrencyShock", "CommodityShock",
    "ElectionOutcome", "InterestRateDecision", "InflationReport",
    "EmploymentReport", "GDPReport"
]

# Define the controlled vocabulary for relationship types
relationship_types = [
    "hasRecommendation", "announcesPolicy", "intervenesIn", "acquires",
    "employs", "hasRole", "produces", "listedOn", "belongsToSector",
    "basedIn", "hasSubsidiary", "isCompetitorOf", "isAffectedByEvent",
    "announcedEvent", "owns"
]

# Define the sentiment values based on the example output
sentiment_values = ["Positive", "Negative", "Neutral"]

# Construct the structured schema based on the updated prompt's output format
schema = Schema(
    type=Type.ARRAY,
    description="A flat JSON array with one object per extracted triplet from a list of financial news article.",
    items=Schema(
        type=Type.OBJECT,
        description="Represents a single semantic relationship (Head, Relation, Tail).",
        required=["H", "R", "T", "ID", "S"],
        properties={
            "H": Schema(
                type=Type.ARRAY,
                description=(
                    "The head (subject) of the relationship triplet. "
                    "Structure: [head_text, head_type, head_canonical]. "
                    f"The 'head_type' must be one of: {', '.join(entity_types)}."
                ),
                min_items=3,
                max_items=3,
                items=Schema(type=Type.STRING)
            ),
            "R": Schema(
                type=Type.STRING,
                description="The relationship type connecting the head and tail.",
                enum=relationship_types
            ),
            "T": Schema(
                type=Type.ARRAY,
                description=(
                    "The tail (object) of the relationship triplet. "
                    "Structure: [tail_text, tail_type, tail_canonical]. "
                    f"The 'tail_type' must be one of: {', '.join(entity_types)}. "
                    f"If 'tail_type' is EVENT, 'tail_canonical' should be one of: {', '.join(event_subtypes)}."
                ),
                min_items=3,
                max_items=3,
                items=Schema(type=Type.STRING)
            ),
            "ID": Schema(
                type=Type.STRING,
                description="The unique identifier (UUID) of the source article, exactly as provided in the input."
            ),
            "S": Schema(
                type=Type.STRING,
                description="The sentiment of the relationship.",
                enum=sentiment_values
            )
        }
    )
)

prompt = """
You are a financial analyst specializing in extracting structured knowledge from news articles for use in financial knowledge graphs.

Your task is to extract **entities** and **relationships** from the given list of articles, using a controlled vocabulary and a standardized schema to support entity disambiguation and graph construction.

Follow the guidelines and return the result in the specified output format. 
 - **Analyze** the article content to identify relevant entities and the relationships between them.
 - **SKIP ARTICLES that contain personal stories and are not relevant**.
 - **DO NOT** return triplets with the same **head and tail** e.g. Utilities belongsToSector Utilities
 - **DO NOT** return triplets with empty head or tail**.
 - **FOLLOW THE GUIDELINES** below for the canonical names. DO NOT RETURN EMPT
 - **RETURN ONLY INFORMATION THAT IS PRESENT IN THE ARTICLE**. DO NOT ADD ANYTHING EXTRA except the canonical names.

### Entity Type Descriptions:
  - **COMPANY**: Public or private legal entity. Use canonical names where possible (e.g., "Alphabet Inc." not "Google"). Canonical name source should Wikidata (P31=Q4830453), OpenCorporates, LEI Registry (GLEIF)
  - **PERSON**: Individual (typically executives or board members), including title or role if available. Canonical name source Wikidata (P31=Q5), professional directories  
  - **PRODUCT**: Financial or commercial product or service. Use canonical name + company if needed. Canonical name source Wikidata 
  - **FINANCIAL_INSTRUMENT**: Equities, bonds, derivatives, etc. Use ticker when available + company. Canonical name source exchanges directories, yahoo, Bloomberg, Refinitiv. 
  - **SECTOR**: Use a restricted GICS or Yahoo Finance sector (e.g., "Technology", "Healthcare").  Canonical name source GICS (11 sectors)
  - **EVENT**: Any corporate, financial event or political event. 
  - **LOCATION**: City or country or geopolitical region e.g Germany (DE). Canonical name source ISO 3166, GADM, Wikidata  
  - **EXCHANGE**: Stock exchange (e.g., "NASDAQ", "NYSE"). Canonical name source Wikidata (`P31=Q44782`), exchange directories  
  - **ROLE**: Executive/Board roles (e.g., "CEO", "CFO"). Canonical name Source Wikidata (`P39=position held`), corporate disclosures  

EXAMPLE INPUT: [{
  "headline": "How investors can protect themselves from Trump’s tariffs with bonds",
  "content_summary": "“There’s no reason to expose your investors to lower-quality credit,” said Adam Abbas at Harris Oakmark.\nHow are debt investors positioned for Trump’s tariff negotiations?\n\nA previous version of this article contained a misspelling of Adam Abbas’s name.\n\nStocks may downplay President Donald Trump’s tactics on negotiating tariffs, but the bond market has been bracing for worse days ahead.",
  "uuid": 893216957
},]

EXPECTED OUTPUT: [
  {
    "H": [
      "Harris Oakmark",
      "COMPANY",
      "Harris Associates",
    ],
    "R": "employs",
    "T": [
      "Adam Abbas",
      "PERSON",
      "Adam Abbas",
    ],
    "ID": "893216957",
    "S": "Neutral"
  },
  {
    "H": [
      "United States",
      "LOCATION",
      "United States",
    ],
    "R": "announcesPolicy",
    "T": [
      "tariffs",
      "EVENT",
      "TradeSanctions",
    ],
    "ID": "893216957",
    "S": "Negative"
  },
  {
    "H": [
      "Adam Abbas",
      "PERSON",
      "Adam Abbas",
    ],
    "R": "hasRecommendation",
    "T": [
      "lower-quality credit",
      "FINANCIAL_INSTRUMENT",
      "Low-Quality Credit",
    ],
    "ID": "893216957",
    "S": "Negative"
  }
]
"""


class KGExtractor(BaseDataHandler):

    def __init__(self, start_date: datetime.date = None, end_date: datetime.date = None, redo: bool = False,
                 model: str = 'gemini-2.5-flash'):
        super().__init__(data_dir=f'raw_article_triplets/{re.sub(r'\W+', "_", model)}',
                         metadata_dir=f'raw_article_triplets/{re.sub(r'\W+', "_", model)}')
        self.data_manager = ArticlePreprocessor(start_date=start_date, end_date=end_date)
        self.start_date = start_date
        self.end_date = end_date
        self.redo = redo
        self.model = model
        self.client = genai.Client(api_key=API_KEY)

    def load_articles(self, date: datetime.date) -> Optional[str]:
        articles = self.data_manager.load(start_date=date, end_date=date, output_format='str')
        if not articles:
            self.logger.warning(f"No articles found for date {date}. You need to run the ArticleLoader first.")
            return
        articles = articles[0]
        no_of_articles = len(re.findall(r'headline', articles))
        self.logger.info(f"Loaded {no_of_articles} articles for date {date}.")
        return articles

    def get_available_dates(self) -> List[datetime.date]:
        """Get a list of available dates for which articles are loaded."""
        paths = self.root.glob('*.json')
        dates = []
        for path in paths:
            try:
                date_str = path.stem
                date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
                dates.append(date)
            except ValueError:
                self.logger.warning(f"Skipping invalid date format in file: {path}")
        return sorted(dates)

    def exists(self, date: datetime.date) -> bool:
        """Check if articles exist for the given date."""
        if self.redo:
            self.logger.info(f"Force download is enabled, skipping existence check for date {date}.")
            return False
        return (self.root / f"{date.strftime('%Y%m%d')}.json").exists()

    def run(self, start_date: datetime.date = None, end_date: datetime.date = None):
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        dates = [dt for dt in pd.date_range(start=start_date, end=end_date) if not self.exists(dt)]
        import numpy as np

        for date in dates:
            articles = self.load_articles(date)
            if not articles:
                self.logger.warning(f"No articles found for date {date}. Skipping.")
                continue
            responses = []
            usage_meta_data = {}
            total_triplets = 0
            articles = json.loads(articles)
            query_count = 1
            processed_articles = []

            self.logger.info(f"Processing {len(articles)} articles for date {date}.")

            total_length = np.sum([len(str(article)) for article in articles])
            if total_length > 90000:
                lengths = sorted([(article, len(str(article))) for article in articles], key=lambda x: x[1])
                if total_length // 2 + total_length % 2 < 70000:
                    smallest_length = lengths[:len(articles) // 2 + len(articles) % 2]
                    largest_lengths = sorted(lengths[len(articles) // 2 + len(articles) % 2:], key=lambda x: x[1],
                                             reverse=True)
                    articles_sorted = []
                    for i, article in enumerate(smallest_length):
                        articles_sorted.append(article[0])
                        if len(largest_lengths) > i:
                            articles_sorted.append(largest_lengths[i][0])
                    n = 2
                    assert len(articles_sorted) == len(articles)
                elif total_length // 3 + total_length % 3 < 70000:
                    split0 = lengths[:len(articles) // 3 + len(articles) % 3]
                    split1 = lengths[len(articles) // 3 + len(articles) % 3:2 * len(articles) // 3 + len(articles) % 3]
                    split2 = sorted(lengths[2 * len(articles) // 3 + len(articles) % 3:], key=lambda x: x[1], reverse=True)
                    articles_sorted = []
                    for i, article in enumerate(split0):
                        articles_sorted.append(article[0])
                        if len(split1) > i:
                            articles_sorted.append(split1[i][0])
                        if len(split2) > i:
                            articles_sorted.append(split2[i][0])
                    n = 3
                    assert len(articles_sorted) == len(articles)
                elif total_length // 4 + total_length % 4 < 70000:
                    split0 = lengths[:len(articles) // 4 + len(articles) % 4]
                    split1 = lengths[len(articles) // 4 + len(articles) % 4:2 * len(articles) // 4 + len(articles) % 4]
                    split2 = sorted(lengths[2 * len(articles) // 4 + len(articles) % 4:3 * len(articles) // 4], key=lambda x: x[1], reverse=True)
                    split3 = sorted(lengths[3 * len(articles) // 4:], key=lambda x: x[1], reverse=True)
                    articles_sorted = []
                    for i, article in enumerate(split0):
                        articles_sorted.append(article[0])
                        if len(split1) > i:
                            articles_sorted.append(split1[i][0])
                        if len(split2) > i:
                            articles_sorted.append(split2[i][0])
                        if len(split3) > i:
                            articles_sorted.append(split3[i][0])
                    n = 4
                    assert len(articles_sorted) == len(articles)
                else:
                    articles_sorted = articles
                    n = 5
            else:
                articles_sorted = articles
                n = 1
            for i in range(n):
                end = min(len(articles_sorted)//n * (i+1) + len(articles_sorted) % n, len(articles))
                articles_chunk = articles_sorted[len(articles)//n * i + int(bool(i)) * len(articles_sorted) % n :end]
                try:
                    self.logger.info(f'Attempting to process {len(articles_chunk)} articles for date {date} of length {len(str(articles_chunk))}.')
                    input_text = f"INPUT: {json.dumps(articles_chunk)}\nOUTPUT:"
                    response = self.client.models.generate_content(
                        model=self.model,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            system_instruction=prompt,
                            response_mime_type="application/json",
                            response_schema=schema,
                            candidate_count=1,
                            thinking_config=types.ThinkingConfig(thinking_budget=9000),
                        ),
                        contents=input_text
                    )
                    responses.extend(json.loads(response.text))
                    triplets = len(json.loads(response.text))
                    total_triplets += triplets
                    usage_meta_data[f'query_{query_count}'] = dict({'triplets_extracted': triplets,
                                                              'articles_processed': len(articles_chunk)},
                                                             **response.usage_metadata.to_json_dict())
                    query_count = query_count + 1

                    processed_articles.extend(articles_chunk)
                except Exception as e:
                    self.logger.warning(f"Input was too long for {date}. Splitting into two chunks.")
                    chunk1 = articles_chunk[:len(articles_chunk) // 2 + len(articles_chunk) % 2]
                    chunk2 = articles_chunk[len(articles_chunk) // 2 + len(articles_chunk) % 2:]
                    for chunk in (chunk1, chunk2):
                        self.logger.info(f'Attempting to process {len(chunk)} articles for date {date} of length {len(str(chunk))}.')

                        input_text = f"INPUT: {json.dumps(chunk)}\nOUTPUT:"
                        response = self.client.models.generate_content(
                            model=self.model,
                            config=types.GenerateContentConfig(
                                temperature=0.3,
                                system_instruction=prompt,
                                response_mime_type="application/json",
                                response_schema=schema,
                                candidate_count=1,
                                thinking_config=types.ThinkingConfig(thinking_budget=9000),
                            ),
                            contents=input_text
                        )
                        responses.extend(json.loads(response.text))
                        triplets = len(json.loads(response.text))
                        total_triplets += triplets
                        usage_meta_data[f'query_{query_count}'] = dict({'triplets_extracted': triplets,
                                                                  'articles_processed': len(chunk)},
                                                                 **response.usage_metadata.to_json_dict())
                        query_count = query_count + 1
                        processed_articles.extend(chunk)

            if len(processed_articles) != len(articles):
                print('AAAAAAAAAA')

            path = self.root / f"{date.strftime('%Y%m%d')}.json"
            with open(path, "w") as f:
                f.write(json.dumps(responses, indent=2, ensure_ascii=True))
            self.logger.info(f"Response received for date {date}. Triplets extracted: {total_triplets} from "
                             f"{len(articles)} articles. Output saved to {path}.")
            metadata_path = self.metadata_root / f"{date.strftime('%Y%m%d')}.json"
            metadata = dict({'extraction_date': datetime.date.today().strftime('%Y-%m-%d'),
                             'triplets_extracted': total_triplets,
                             'articles_processed': len(articles),
                             'date': date.strftime('%Y-%m-%d')},
                            **usage_meta_data)
            with open(metadata_path, "w") as f:
                f.write(json.dumps(metadata, indent=2, ensure_ascii=True))
            self.logger.info(f"Metadata saved to {metadata_path}.")

    def run_old(self, start_date: datetime.date = None, end_date: datetime.date = None):
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        dates = [dt for dt in pd.date_range(start=start_date, end=end_date) if not self.exists(dt)]

        for date in dates:
            articles = self.load_articles(date)
            if not articles:
                self.logger.warning(f"No articles found for date {date}. Skipping.")
                continue
            responses = []
            usage_meta_data = {}
            total_triplets = 0
            articles = json.loads(articles)
            start_i = 0
            i = 0
            query_count = 1
            articles_chunk = []
            processed_articles = []
            self.logger.info(f"Processing {len(articles)} articles for date {date}.")
            while i <= len(articles) - 1:
                articles_chunk.append(articles[i])
                i = i + 1

                if (len(str(articles[start_i: i])) < 81000) and (i < len(articles)):
                    pass
                else:
                    if len(str(articles_chunk)) >= 99000:
                        i = i - 1
                        articles_chunk = articles_chunk[:-1]
                    try:
                        self.logger.info(f'Attempting to process {len(articles_chunk)} articles for date {date} of length {len(str(articles_chunk))}.')
                        input_text = f"INPUT: {json.dumps(articles_chunk)}\nOUTPUT:"
                        response = self.client.models.generate_content(
                            model=self.model,
                            config=types.GenerateContentConfig(
                                temperature=0.3,
                                system_instruction=prompt,
                                response_mime_type="application/json",
                                response_schema=schema,
                                candidate_count=1,
                                thinking_config=types.ThinkingConfig(thinking_budget=9000),
                            ),
                            contents=input_text
                        )
                        responses.extend(json.loads(response.text))
                        triplets = len(json.loads(response.text))
                        total_triplets += triplets
                        usage_meta_data[f'query_{query_count}'] = dict({'triplets_extracted': triplets,
                                                                  'articles_processed': len(articles_chunk)},
                                                                 **response.usage_metadata.to_json_dict())
                        start_i = i
                        query_count = query_count + 1
                        processed_articles.extend(articles_chunk)
                        articles_chunk = []
                    except Exception as e:
                        if len(articles_chunk) > 5:
                            articles_chunk = articles_chunk[:-4]
                            i -= 4
                        else:
                            articles_chunk = articles_chunk[:-2]
                            i -= 2

                        self.logger.warning(f"Input was too long for {date}. Splitting into smaller chunks.")
                        self.logger.info(f'Attempting to process {len(articles_chunk)} articles for date {date} of length {len(str(articles_chunk))}.')

                        input_text = f"INPUT: {json.dumps(articles_chunk)}\nOUTPUT:"
                        response = self.client.models.generate_content(
                            model=self.model,
                            config=types.GenerateContentConfig(
                                temperature=0.3,
                                system_instruction=prompt,
                                response_mime_type="application/json",
                                response_schema=schema,
                                candidate_count=1,
                                thinking_config=types.ThinkingConfig(thinking_budget=9000),
                            ),
                            contents=input_text
                        )
                        responses.extend(json.loads(response.text))
                        triplets = len(json.loads(response.text))
                        total_triplets += triplets
                        usage_meta_data[f'query_{query_count}'] = dict({'triplets_extracted': triplets,
                                                                  'articles_processed': len(articles_chunk)},
                                                                 **response.usage_metadata.to_json_dict())

                        start_i = i
                        query_count = query_count + 1
                        processed_articles.extend(articles_chunk)
                        articles_chunk = []

            if len(processed_articles) != len(articles):
                print('AAAAAAAAAA')

            path = self.root / f"{date.strftime('%Y%m%d')}.json"
            with open(path, "w") as f:
                f.write(json.dumps(responses, indent=2, ensure_ascii=True))
            self.logger.info(f"Response received for date {date}. Triplets extracted: {total_triplets} from "
                             f"{len(articles)} articles. Output saved to {path}.")
            metadata_path = self.metadata_root / f"{date.strftime('%Y%m%d')}.json"
            metadata = dict({'extraction_date': datetime.date.today().strftime('%Y-%m-%d'),
                             'triplets_extracted': total_triplets,
                             'articles_processed': len(articles),
                             'date': date.strftime('%Y-%m-%d')},
                            **usage_meta_data)
            with open(metadata_path, "w") as f:
                f.write(json.dumps(metadata, indent=2, ensure_ascii=True))
            self.logger.info(f"Metadata saved to {metadata_path}.")

    def download(self) -> Any:
        self.logger.info(f"Not implemented for {self.__class__.__name__}.")

    def _do_load(self, path: Path) -> Any:
        """
        Load the triplets from the specified path.
        """
        if not path.exists():
            self.logger.warning(f"Path {path} does not exist. Returning empty list.")
            return []
        with open(path, 'r') as f:
            return json.load(f)

    def load(self, start_date: datetime.date, end_date: datetime.date) -> Any:
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        dates = [dt for dt in pd.date_range(start=start_date, end=end_date) if self.exists(dt)]
        paths = [self.root / f"{date.strftime('%Y%m%d')}.json" for date in dates]
        paths = sorted(paths, key=lambda x: datetime.datetime.strptime(x.stem, '%Y%m%d').date())
        data = dict(H=[], R=[], T=[], date=[])
        sentiment_dict = {'Positive': '_pos', 'Neutral': '', 'Negative': '_neg'}
        for i, path in enumerate(paths):
            result = json.loads(path.read_text())
            for d in result:
                data['H'].append(d['H'][2].upper())
                data['R'].append(d['R'] + sentiment_dict[d['S']])
                data['T'].append(d['T'][2].upper())
                data['date'].append(datetime.datetime.strptime(path.stem, '%Y%m%d').date())
        return pd.DataFrame(data, columns=['H', 'R', 'T', 'D', 'date'])


def run(path):
    client = genai.Client(api_key=None)

    with open(path) as f:
        articles = json.load(f)
    input_text = f"INPUT: {json.dumps(articles[0:60])}\nOUTPUT:"

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            temperature=0.3,
            system_instruction=prompt,
            response_mime_type="application/json",
            response_schema=schema,
            candidate_count=1,
            thinking_config=types.ThinkingConfig(thinking_budget=9000),
        ),
        contents=input_text
    )

    outdir = Path("./output")
    outdir.mkdir(exist_ok=True)
    out_path = outdir / f"flash_{Path(path).stem}.json"
    with open(out_path, "w") as f:
        f.write(response.text)

    print(f"Output written to {out_path}")
    print(f"Usage: {response.usage_metadata}")


if __name__ == '__main__':
    kge = KGExtractor(start_date=datetime.date(2021, 1, 1), end_date=datetime.date(2022, 12, 31), redo=False,
                      model='gemini-2.5-flash')
    kge.run(start_date=datetime.date(2021, 1, 1), end_date=datetime.date(2022, 12, 31))
