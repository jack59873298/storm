"""
STORM Wiki pipeline powered by DeepSeek (via LiteLLM) and Tavily search.
Set up keys in secrets.toml:
    DEEPSEEK_API_KEY="your_key"
    TAVILY_API_KEY="your_key"
"""

import os
import re
import logging
from argparse import ArgumentParser

from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import (
    YouRM,
    BingSearch,
    BraveRM,
    SerperRM,
    DuckDuckGoSearchRM,
    TavilySearchRM,
    SearXNG,
)
from knowledge_storm.utils import load_api_key


def sanitize_topic(topic):
    topic = topic.replace(" ", "_")
    topic = re.sub(r"[^a-zA-Z0-9_-]", "", topic)
    if not topic:
        topic = "unnamed_topic"
    return topic


def main(args):
    load_api_key(toml_file_path="secrets.toml")

    # LiteLLM uses DEEPSEEK_API_KEY env var automatically for deepseek/ prefixed models
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

    lm_configs = STORMWikiLMConfigs()

    deepseek_kwargs = {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
    }

    # Use deepseek/ prefix for litellm routing
    model_name = f"deepseek/{args.model}"

    conv_simulator_lm = LitellmModel(model=model_name, max_tokens=500, **deepseek_kwargs)
    question_asker_lm = LitellmModel(model=model_name, max_tokens=500, **deepseek_kwargs)
    outline_gen_lm = LitellmModel(model=model_name, max_tokens=400, **deepseek_kwargs)
    article_gen_lm = LitellmModel(model=model_name, max_tokens=700, **deepseek_kwargs)
    article_polish_lm = LitellmModel(model=model_name, max_tokens=4000, **deepseek_kwargs)

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)
    lm_configs.set_article_polish_lm(article_polish_lm)

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    match args.retriever:
        case "bing":
            rm = BingSearch(bing_search_api=os.getenv("BING_SEARCH_API_KEY"), k=engine_args.search_top_k)
        case "you":
            rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=engine_args.search_top_k)
        case "brave":
            rm = BraveRM(brave_search_api_key=os.getenv("BRAVE_API_KEY"), k=engine_args.search_top_k)
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(k=engine_args.search_top_k, safe_search="On", region="us-en")
        case "serper":
            rm = SerperRM(serper_search_api_key=os.getenv("SERPER_API_KEY"), query_params={"autocorrect": True, "num": 10, "page": 1})
        case "tavily":
            rm = TavilySearchRM(tavily_search_api_key=os.getenv("TAVILY_API_KEY"), k=engine_args.search_top_k, include_raw_content=True)
        case "searxng":
            rm = SearXNG(searxng_api_key=os.getenv("SEARXNG_API_KEY"), k=engine_args.search_top_k)
        case _:
            raise ValueError(f'Invalid retriever: {args.retriever}')

    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    topic = input("Topic: ")
    sanitized_topic = sanitize_topic(topic)

    try:
        runner.run(
            topic=sanitized_topic,
            do_research=args.do_research,
            do_generate_outline=args.do_generate_outline,
            do_generate_article=args.do_generate_article,
            do_polish_article=args.do_polish_article,
            remove_duplicate=args.remove_duplicate,
        )
        runner.post_run()
        runner.summary()
    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        raise

    output_dir = os.path.join(args.output_dir, sanitized_topic)
    print("\n" + "=" * 60)
    print("STORM COMPLETE")
    print("=" * 60)
    print(f"Topic:   {topic}")
    print(f"Output:  {os.path.abspath(output_dir)}")
    print(f"\nKey files:")
    polished = os.path.join(output_dir, "storm_gen_article_polished.txt")
    article = os.path.join(output_dir, "storm_gen_article.txt")
    if os.path.exists(polished):
        print(f"  Article:  storm_gen_article_polished.txt")
    elif os.path.exists(article):
        print(f"  Article:  storm_gen_article.txt")
    print(f"  Outline:  storm_gen_outline.txt")
    print(f"  Sources:  url_to_info.json")
    print(f"  Research: conversation_log.json")
    print("=" * 60)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./results/deepseek")
    parser.add_argument("--max-thread-num", type=int, default=3)
    parser.add_argument("--retriever", type=str, choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"], default="tavily")
    parser.add_argument("--model", type=str, default="deepseek-chat", help='DeepSeek model name (e.g. deepseek-chat, deepseek-reasoner)')
    parser.add_argument("--do-research", action="store_true")
    parser.add_argument("--do-generate-outline", action="store_true")
    parser.add_argument("--do-generate-article", action="store_true")
    parser.add_argument("--do-polish-article", action="store_true")
    parser.add_argument("--max-conv-turn", type=int, default=3)
    parser.add_argument("--max-perspective", type=int, default=3)
    parser.add_argument("--search-top-k", type=int, default=3)
    parser.add_argument("--retrieve-top-k", type=int, default=3)
    parser.add_argument("--remove-duplicate", action="store_true")

    main(parser.parse_args())
