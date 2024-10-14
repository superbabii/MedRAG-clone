import json
import random
from src.medrag import MedRAG

# Load benchmark data
with open("benchmark.json", "r") as file:
    benchmark_data = json.load(file)

# Select 5 random samples from the benchmark
random_questions = random.sample(list(benchmark_data.items()), 5)

# Initialize the MedRAG model
medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")

# Prepare a list to hold comparison results
results = []

# Loop through the selected questions and generate answers
for question_id, data in random_questions:
    question = data["question"]
    options = data["options"]
    correct_answer = data["answer"]

    # Get model's prediction
    answer, snippets, scores = medrag.answer(question=question, options=options, k=5)

    # Compare with the correct answer
    result = {
        "question_id": question_id,
        "question": question,
        "options": options,
        "model_answer": answer['answer_choice'],
        "correct_answer": correct_answer,
        "is_correct": answer['answer_choice'] == correct_answer
    }

    results.append(result)

# Print results for comparison
for result in results:
    print(f"Question ID: {result['question_id']}")
    print(f"Question: {result['question']}")
    print(f"Options: {result['options']}")
    print(f"Model Answer: {result['model_answer']}")
    print(f"Correct Answer: {result['correct_answer']}")
    print(f"Correct: {result['is_correct']}")
    print("-" * 50)

# Calculate accuracy
accuracy = sum([1 for result in results if result['is_correct']]) / len(results)
print(f"Accuracy: {accuracy * 100}%")
