import os
import sys
import gradio as gr
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompt import system_prompt


load_dotenv()
os.environ["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")


embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



def build_conversation_context(state, max_turns=8):
    if not state:
        return ""
    turns = state[-max_turns:]
    return "\n\n".join([f"User: {u}\nAssistant: {a}" for u, a in turns])



def chatbot_fn(user_message, chat_history, state):
    chat_history = chat_history or []
    state = state or []
    if chat_history and isinstance(chat_history[0], tuple):
        chat_history = [
            {"role": "user", "content": user}
            for user, assistant in chat_history
        ] + [
            {"role": "assistant", "content": assistant}
            for user, assistant in chat_history
        ]
    conversation_context = build_conversation_context(state, max_turns=12)
    combined_input = f"{conversation_context}\n\nUser: {user_message}\nAssistant: Continue the conversation, answer the user's question, and use the retrieved documents as needed." if conversation_context else user_message
    try:
        result = rag_chain.invoke({"input": combined_input})
        assistant_reply = result.get("answer") if isinstance(result, dict) else str(result)
        if not assistant_reply and isinstance(result, dict):
            assistant_reply = result.get("output_text") or result.get("text") or str(result)
    except Exception as e:
        print("Error in chatbot_fn:", e, file=sys.stderr)
        assistant_reply = "Logs"
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": assistant_reply})
    state.append((user_message, assistant_reply))
    max_state_turns = 40
    if len(state) > max_state_turns:
        state = state[-max_state_turns:]
    return chat_history, state



def reset_chat():
    return [], []

# --- Gradio UI (Blocks) ---

with gr.Blocks() as demo:
    gr.Markdown("# Health_Care_ChatBot")
    chatbot = gr.Chatbot(label="Health_Care_ChatBot", type="messages", value=[])
    msg = gr.Textbox(show_label=False, placeholder="Enter")
    state = gr.State([])
    submit = msg.submit(chatbot_fn, inputs=[msg, chatbot, state], outputs=[chatbot, state])
    clear = gr.Button("Clear Conversation")
    clear.click(fn=reset_chat, outputs=[chatbot, state])
    msg.submit(lambda x: gr.update(value=''), [],[msg])

# --- Launch (compatible with HF Spaces) ---
if __name__ == "__main__":
    # On Hugging Face Spaces, simply calling demo.launch() works.
    # To be explicit about binding (helpful in some environments):
    demo.launch()
