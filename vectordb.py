import os
import chromadb
import json
import time
import torch

from transformers import AutoModel,AutoTokenizer
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("upskyy/bge-m3-korean")

class EmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        # input은 문장들의 리스트 형태로 들어옵니다.
        embeddings = self.model.encode(input)
        return embeddings  

client = chromadb.PersistentClient(path = "./chroma_db")

embedding_function = EmbeddingFunction(model)




collection = client.get_or_create_collection(name="test_vector"
	,embedding_function = embedding_function
	,metadata={"hnsw:space": "cosine"})




#데이터 추가
def addData(title, newMetadatas,contents) :
	try : 
		collection.upsert(
			documents = contents ,
			metadatas = newMetadatas,
			ids = title)
		return True
	except :
		return False

#데이터 삭제
def deleteData(title) : 
	collection.delete(ids = title)
	result = collection.get(ids=title)
	return bool(result['ids'])

#데이터 조회
def searchData(question) :	
	
	start = time.time()
	time.sleep(1)
	result = collection.query(
    		query_texts=[question],
    		n_results=3
		)
	print(f"{time.time()-start:.4f} sec") # 종료와 함께 수행시간 출력
	
	filtered_results = [
    	(title,doc, score) for title,doc, score in zip(result["ids"][0],result["documents"][0], result["distances"][0]) if score <= 0.5]	
	
	print(filtered_results)
	
	return filtered_results

#조건문 데이터 탐색



def searchRules(depart, field,name) :
	#유사도 탐색이 아닌 일반 데이터 필터링
	start = time.time()
	time.sleep(1)
	print("!")
	if depart is None and field is None :
		result = collection.get(
			where = {"title" : name})
	else : 
		result = collection.get(
			where = {"title" : name})
	print(f"{time.time()-start:.4f} sec") # 종료와 함께 수행[>


	# result에서 ids와 documents 값만 추출
	ids = result.get('ids', [])
	documents = result.get('documents', [])
	metadatas = result.get('metadatas',[])

	return_data = {
	'ids': ids,
	'documents': documents,
	'metadatas': metadatas
	}

	return return_data


# 폴더 내 파일 읽기
def createVectorDb() :
	file_path = "text3.txt"  # 파일이 있는 폴더 경로	
	titles=[]
	summaries = []
	count = 0

	# 파일 읽기 및 데이터 처리
	with open(file_path, "r", encoding="utf-8") as text_file:
		content = text_file.read()  # JSON 데이터 로드
		sentences = content.split("!")
		sentences = [sentence.strip() for sentence in sentences if sentence. strip()]
		# data 객체 처리
		for sentence in enumerate(sentences) :

			parts = sentence[1].split("$", 1)
			title,summary_text = parts
           		# 데이터 추가
			
			title = title.strip()
			summary_text = summary_text.strip()
			
			titles.append(title)
			summaries.append(summary_text)
		
			#메타데이터
			metadata = [{"field" : "부속기관", "depart" : "도서관","title" : "도서관규정"}]*len(summaries)
			
			count +=1
	
			if count > 5000 : 
				# 벡터 DB에 저장
				collection.add(
					documents=summaries,  # 본문
					ids=titles,	      # 고유아이디
					metadatas = metadata
				)
				count = 0
	# 남은 데이터도 저장
	if count < 5000 :
		collection.add(
    			documents=summaries,    
    			ids=titles,
			metadatas = metadata
			)       

def deleteVectorDb():
	client.delete_collection(name = "test_vector")
	collections = client.list_collections()
	print("Remaining collections:", collections)


#createVectorDb()
#deleteVectorDb()
#searchData("교원 비밀 누설 관련")
#searchData("도서관 구입 예산")
#searchRules("depart","도서관")
