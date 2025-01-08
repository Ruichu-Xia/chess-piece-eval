import io
import jsonlines

import zstandard as zstd
from loguru import logger

from config import data_settings


def process_eval_zst(compressed_file, line_limit=None):
    dctx = zstd.ZstdDecompressor()
    with open(compressed_file, 'rb') as ifh:
        with dctx.stream_reader(ifh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            json_reader = jsonlines.Reader(text_stream)
            processed_data = []
            count = 0
            logger.info(f"Processing {compressed_file}")
            for obj in json_reader:
                fen = obj['fen']
                pvs = obj['evals'][0]['pvs'][0]
                if 'cp' in pvs:
                    eval = pvs['cp'] / 100
                    eval = max(-data_settings.max_eval,
                               min(data_settings.max_eval, eval))
                elif 'mate' in pvs:
                    if pvs['mate'] > 0:
                        eval = data_settings.max_eval
                    else:
                        eval = -data_settings.max_eval

                processed_data.append((fen, eval))
                count += 1
                if count % 1000000 == 0:
                    logger.info(f"Processed {count} lines")
                if line_limit and count >= line_limit:
                    break
        return processed_data
