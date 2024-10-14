from src.medrag import MedRAG

# Define the question and answer options
question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

# Step 1: Using MedRAG with no RAG (direct LLM query)
# cot = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=False)
# answer, _, _ = cot.answer(question=question, options=options)
# print(f"Final answer in JSON with rationale: {answer}")
# Expected output:
# {
#   "step_by_step_thinking": "Compression of the facial nerve at the stylomastoid foramen will affect the function of the facial nerve. The facial nerve is responsible for innervating the muscles of facial expression, including those involved in smiling, frowning, and closing the eyes. It also carries taste sensation from the anterior two-thirds of the tongue. Additionally, the facial nerve controls tear production (lacrimation) and salivation. Therefore, compression of the facial nerve at the stylomastoid foramen will cause paralysis of the facial muscles (A), loss of taste (B), lacrimation (C), and decreased salivation (D).",
#   "answer_choice": "D"
# }

# Step 2: Using MedRAG with RAG enabled and textbook corpus
medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")
answer, snippets, scores = medrag.answer(question=question, options=options, k=2)
print(f"Final answer in JSON with rationale: {answer}")
# Expected output:
# {
#   "step_by_step_thinking": "A lesion causing compression of the facial nerve at the stylomastoid foramen will result in paralysis of the facial muscles. Loss of taste, lacrimation, and decreased salivation are not specifically mentioned in relation to a lesion at the stylomastoid foramen.",
#   "answer_choice": "A"
# }

# Step 3: MedRAG with pre-determined snippets
# snippets = [
#     {
#         'id': 'InternalMed_Harrison_30037', 
#         'title': 'InternalMed_Harrison', 
#         'content': 'On side of lesion Horizontal and vertical nystagmus, vertigo, nausea, vomiting, oscillopsia: Vestibular nerve or nucleus Facial paralysis: Seventh nerve Paralysis of conjugate gaze to side of lesion: Center for conjugate lateral gaze Deafness, tinnitus: Auditory nerve or cochlear nucleus Ataxia: Middle cerebellar peduncle and cerebellar hemisphere Impaired sensation over face: Descending tract and nucleus fifth nerve On side opposite lesion Impaired pain and thermal sense over one-half the body (may include face): Spinothalamic tract Although atheromatous disease rarely narrows the second and third segments of the vertebral artery, this region is subject to dissection, fibromuscular dysplasia, and, rarely, encroachment by osteophytic spurs within the vertebral foramina.'
#     }
# ]
# answer, _, _ = medrag.answer(question=question, options=options, snippets=snippets)
# print(f"Final answer with pre-determined snippets: {answer}")

# # Step 4: MedRAG with pre-determined snippet IDs
# snippet_ids = [{"id": s["id"]} for s in snippets]
# answer, snippets, _ = medrag.answer(question=question, options=options, snippets_ids=snippet_ids)
# print(f"Final answer with pre-determined snippet IDs: {answer}")

# # Step 5: Using MedRAG with RAG and cached corpus
# medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks", corpus_cache=True)
# answer, snippets, scores = medrag.answer(question=question, options=options, k=32)
# print(f"Final answer with corpus caching: {answer}")

# # Step 6: Using MedRAG with follow-up questions and multi-round querying
# medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, follow_up=True, retriever_name="MedCPT", corpus_name="Textbooks", corpus_cache=True)
# answer, history = medrag.answer(question=question, options=options, k=32, n_rounds=4, n_queries=3)
# print(f"Final answer in JSON: {answer}")  # e.g., {'answer': 'A'}
# print(f"Raw answer with analysis: {history[-3]}")
# # Follow-up queries generated in the process:
# print(f"Follow-up queries generated: {[item.split('Answer: ')[0].strip() for item in history[-4]['content'].split('Query: ')[1:]]}")
