import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

Vec_DB = FAISS.load_local(
    folder_path='faiss_index',
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)
llm = ChatOpenAI()


def query_similar_documents(query: str):
    documents = Vec_DB.similarity_search(query)
    return [doc.page_content for doc in documents]


def sales_chat(message, history):
    print(f'[message]{message}')
    print(f'[history]{history}')
    related_docs = query_similar_documents(message)

    if len(related_docs) == 0:
        resp = llm.invoke(input=message)
    else:
        prompt = (f'从检索到的知识中回答用户的问题, 仅回答问题，不要出现知识来源\n'
                  f'# 检索的知识\n{related_docs}\n\n'
                  f'# 用户问题\n{message}')
        resp = llm.invoke(input=prompt)

    return resp.content


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title='汽车销售',
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name='0.0.0.0')


if __name__ == "__main__":
    # 启动 Gradio 服务
    launch_gradio()
