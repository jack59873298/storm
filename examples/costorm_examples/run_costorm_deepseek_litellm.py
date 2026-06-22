"""
Co-STORM pipeline powered by DeepSeek (via LiteLLM) and Tavily search.
Set up keys in secrets.toml:
    DEEPSEEK_API_KEY="your_key"
    TAVILY_API_KEY="your_key"

Interactive research session — you can observe AI agents discuss or inject
your own questions to steer the conversation.

Output:
    args.output_dir/
        report.md              # Final article
        instance_dump.json     # Full session data
        log.json               # LLM call logs
"""

import os
import json
from argparse import ArgumentParser
from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs,
    RunnerArgument,
    CoStormRunner,
)
from knowledge_storm.collaborative_storm.modules.callback import (
    LocalConsolePrintCallBackHandler,
)
from knowledge_storm.lm import LitellmModel
from knowledge_storm.logging_wrapper import LoggingWrapper
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


def main(args):
    load_api_key(toml_file_path="secrets.toml")
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")

    lm_config = CollaborativeStormLMConfigs()

    deepseek_kwargs = {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
    }

    model_name = f"deepseek/{args.model}"

    question_answering_lm = LitellmModel(model=model_name, max_tokens=1000, **deepseek_kwargs)
    discourse_manage_lm = LitellmModel(model=model_name, max_tokens=500, **deepseek_kwargs)
    utterance_polishing_lm = LitellmModel(model=model_name, max_tokens=2000, **deepseek_kwargs)
    warmstart_outline_gen_lm = LitellmModel(model=model_name, max_tokens=500, **deepseek_kwargs)
    question_asking_lm = LitellmModel(model=model_name, max_tokens=300, **deepseek_kwargs)
    knowledge_base_lm = LitellmModel(model=model_name, max_tokens=1000, **deepseek_kwargs)

    lm_config.set_question_answering_lm(question_answering_lm)
    lm_config.set_discourse_manage_lm(discourse_manage_lm)
    lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
    lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
    lm_config.set_question_asking_lm(question_asking_lm)
    lm_config.set_knowledge_base_lm(knowledge_base_lm)

    topic = input("Topic: ")
    runner_argument = RunnerArgument(
        topic=topic,
        retrieve_top_k=args.retrieve_top_k,
        max_search_queries=args.max_search_queries,
        total_conv_turn=args.total_conv_turn,
        max_search_thread=args.max_search_thread,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        warmstart_max_num_experts=args.warmstart_max_num_experts,
        warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
        warmstart_max_thread=args.warmstart_max_thread,
        max_thread_num=args.max_thread_num,
        max_num_round_table_experts=args.max_num_round_table_experts,
        moderator_override_N_consecutive_answering_turn=args.moderator_override_N_consecutive_answering_turn,
        node_expansion_trigger_count=args.node_expansion_trigger_count,
    )
    logging_wrapper = LoggingWrapper(lm_config)
    callback_handler = LocalConsolePrintCallBackHandler() if args.enable_log_print else None

    match args.retriever:
        case "bing":
            rm = BingSearch(bing_search_api=os.getenv("BING_SEARCH_API_KEY"), k=runner_argument.retrieve_top_k)
        case "you":
            rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=runner_argument.retrieve_top_k)
        case "brave":
            rm = BraveRM(brave_search_api_key=os.getenv("BRAVE_API_KEY"), k=runner_argument.retrieve_top_k)
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(k=runner_argument.retrieve_top_k, safe_search="On", region="us-en")
        case "serper":
            rm = SerperRM(serper_search_api_key=os.getenv("SERPER_API_KEY"), query_params={"autocorrect": True, "num": 10, "page": 1})
        case "tavily":
            rm = TavilySearchRM(tavily_search_api_key=os.getenv("TAVILY_API_KEY"), k=runner_argument.retrieve_top_k, include_raw_content=True)
        case "searxng":
            rm = SearXNG(searxng_api_key=os.getenv("SEARXNG_API_KEY"), k=runner_argument.retrieve_top_k)
        case _:
            raise ValueError(f'Invalid retriever: {args.retriever}')

    costorm_runner = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=rm,
        callback_handler=callback_handler,
    )

    print("\n[Co-STORM] Warming up — researching initial perspectives...")
    costorm_runner.warm_start()
    print("[Co-STORM] Warm start complete. Entering interactive mode.\n")

    print("=" * 60)
    print("INTERACTIVE MODE")
    print("  Type your question to steer the discussion")
    print("  Press Enter (empty) to observe the next AI turn")
    print("  Type 'done' to generate the final report")
    print("=" * 60 + "\n")

    turn = 0
    while turn < args.total_conv_turn:
        user_input = input(f"[Turn {turn + 1}/{args.total_conv_turn}] You (or Enter to observe, 'done' to finish): ").strip()

        if user_input.lower() == "done":
            break

        if user_input:
            costorm_runner.step(user_utterance=user_input)
            print()
        else:
            conv_turn = costorm_runner.step()
            print(f"**{conv_turn.role}**: {conv_turn.utterance}\n")

        turn += 1

    print("\n[Co-STORM] Generating final report...")
    costorm_runner.knowledge_base.reorganize()
    article = costorm_runner.generate_report()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(article)

    instance_copy = costorm_runner.to_dict()
    with open(os.path.join(args.output_dir, "instance_dump.json"), "w", encoding="utf-8") as f:
        json.dump(instance_copy, f, indent=2)

    log_dump = costorm_runner.dump_logging_and_reset()
    with open(os.path.join(args.output_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log_dump, f, indent=2)

    print("\n" + "=" * 60)
    print("CO-STORM COMPLETE")
    print("=" * 60)
    print(f"Topic:   {topic}")
    print(f"Output:  {os.path.abspath(args.output_dir)}")
    print(f"\nKey files:")
    print(f"  Report:  report.md")
    print(f"  Session: instance_dump.json")
    print(f"  Logs:    log.json")
    print("=" * 60)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./results/co-storm")
    parser.add_argument("--retriever", type=str, choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"], default="tavily")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--max_search_queries", type=int, default=2)
    parser.add_argument("--total_conv_turn", type=int, default=20)
    parser.add_argument("--max_search_thread", type=int, default=5)
    parser.add_argument("--max_search_queries_per_turn", type=int, default=3)
    parser.add_argument("--warmstart_max_num_experts", type=int, default=3)
    parser.add_argument("--warmstart_max_turn_per_experts", type=int, default=2)
    parser.add_argument("--warmstart_max_thread", type=int, default=3)
    parser.add_argument("--max_thread_num", type=int, default=10)
    parser.add_argument("--max_num_round_table_experts", type=int, default=2)
    parser.add_argument("--moderator_override_N_consecutive_answering_turn", type=int, default=3)
    parser.add_argument("--node_expansion_trigger_count", type=int, default=10)
    parser.add_argument("--enable_log_print", action="store_true")

    main(parser.parse_args())
