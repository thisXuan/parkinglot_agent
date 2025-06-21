from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 读取文件
def load_file(file):
    directory_path = '.'
    loader = DirectoryLoader(directory_path, glob=file)
    documents = loader.load()
    return documents

# 对文本进行分片
def chunk_data(data,chunk_size,chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(data)
    return docs

if __name__ == "__main__":
    data = load_file('auto_data.txt')
    chunks = chunk_data(data)
    print(f"将文档分割成 {len(chunks)} 个块")