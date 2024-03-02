import imageio
import os
import base64
import io
import json
import argparse
import ollama
import glob
from tqdm import tqdm
import time
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Tongyi
from langchain_community.document_loaders import (
    TextLoader,
)
from dashscope import MultiModalConversation
import dashscope
from zhipuai import ZhipuAI
dashscope.api_key="/Your/API/KEY"
client = ZhipuAI(api_key="/Your/Zhipu/API/KEY") # 填写您自己的APIKey
load_dotenv()
# Load environment variables

source_directory = 'source_documents'
chunk_size = 500
chunk_overlap = 50
target_source_chunks = 4
os.environ["IMAGEIO_FFMPEG_EXE"] = "/path/to/ffmpeg"

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, help='Path to the video file')
parser.add_argument('--stride', required=True, help='interval to describe the image')
parser.add_argument('--lvm', required=False, help='LVM to parse the image or video')

args = parser.parse_args()

#parse the video with llava into txt if source_documents is empty
def video_parse(video_path):
    reader = imageio.get_reader(video_path, 'ffmpeg')
    meta = reader.get_meta_data()
    try:
        total_frames = meta["n_frames"]  # Access frame count from metadata (if available)
    except KeyError:
        print("Frame count not found in metadata. Counting frames manually...")
        total_frames = 0
        for _ in reader:
            total_frames += 1

    print(f"Number of frames: {total_frames}")

    for i in tqdm(range(total_frames)):

        frame = reader.get_next_data()

        if i % int(args.stride) == 0: 
            # Save the image to a file
            imageio.imsave('temp.png', frame)

            local_file_path1 = 'file:///Users/zhouql1978/dev/qwen/temp.png'

            match args.lvm:
                case "Qwen":
                    messages = [{
                            'role': 'system',
                            'content': [{
                                'text': '你是一位非常出色的侦探，你将帮我观察路上可疑的行人和车辆.'
                            }]
                        }, {
                            'role':
                            'user',
                            'content': [
                                {
                                    'image': local_file_path1
                                },
                                {
                                    'text': '请用中文告诉我图片上描述了什么，重点关注路上和行人和车辆，特别要记下经过的车牌号和车辆颜色'
                                },
                            ]
                        }]
                    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
                    data=json.loads(str(response))
                    result=str(data['output']['choices'][0]['message']['content'][0]['text'])
                case "Zhipu":
                    with open('temp.png', 'rb') as f:
                        image_data = f.read()
 
                    # 将二进制数据编码成Base64格式的字符串
                    encoded_string = base64.b64encode(image_data).decode('utf-8')
                    #print("编码后的图片 Base64 字符串:\n", encoded_string)

                    response = client.chat.completions.create(
                        model="glm-4v",  # 填写需要调用的模型名称
                        messages=[
                        {
                            "role": "user",
                            "content": [
                            {
                                "type": "text",
                                "text": "请用中文告诉我图片上描述了什么，重点关注路上和行人和车辆，特别要记下经过的车牌号和车辆颜色"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url" : encoded_string
                                }
                            }
                            ]
                        }
                        ]
                    )
                    result=str(response.choices[0].message.content)
                case _default:
                    response = ollama.chat(
	                    model="llava",
	                    messages=[
		                    {
			                    'role': 'user',
			                    #'content': 'You are a detective, please describe this image, and particularly focus on the pedestrian and cars, give me more details about the plate number and coloar of the car',
			                    'content': "请用中文告诉我图片上描述了什么，重点关注路上和行人和车辆，特别要记下经过的车牌号和车辆颜色",
                                'images': ['./temp.png']
		                    }
	                    ]
                    )
                    result=str(response['message']['content'])
            #print(str(data['output']['choices'][0]['message']['content'][0]['text']))

            
            with open("source_documents/"+video_path+".txt", "a") as write_file:
                write_file.write("---"*10 + "\n\n")
                write_file.write("Frame:"+str(i) + "\n\n")
                write_file.write(result)
                write_file.flush()
        
    reader.close()

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            
            for j, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def is_dir_empty(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return not os.listdir(path)

# The mail loop
def main():

    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    if is_dir_empty(source_directory):
        video_parse(args.path)

    try:
        os.listdir("faiss_index")
  
    except FileNotFoundError:
        # Create and store locally vectorstore if folder not exit
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
    
        db = FAISS.from_documents(texts, embeddings)
        db.save_local("faiss_index")
        print(f"Ingestion complete! You can now run to query your documents")

    #loading the vectorstore
    db=FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm=Tongyi()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if False else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:            
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


if __name__ == "__main__":
    main()
