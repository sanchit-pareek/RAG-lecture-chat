from backend.rag import load_index, retrieve, call_llm, build_prompt

def main():
    frame, matrix, _ = load_index()
    while True:
        question = input("Ask a question (or 'exit'): ").strip()
        if question.lower() == 'exit':
            break
        if not question:
            continue
        result = retrieve(frame, matrix, question)
        print(call_llm(build_prompt(question, result['contexts'])) if result['contexts']
              else "I don't know based on the provided lectures.")

if __name__ == '__main__':
    main()
