from agents.research_agent import ResearchAgent

def main():
    agent = ResearchAgent()

    print("=" * 60)
    print("       ResearchForge - Autonomous Research Assistant")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Ask a research question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nGoodbye! Thanks for using ResearchForge.")
            break

        if not question:
            continue

        try:
            agent.run(question)
        except Exception as e:
            print(f"\nAn error occurred: {e}")

        print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()