from fastapi import FastAPI,Depends,Path,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prompt import model,answer,startMessage
from vectordb import searchData,searchRules,addData,deleteData

import uvicorn


from models import Base, Test
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

new_model = None
new_processor = None

class Message(BaseModel):
	text : str

class Rule(BaseModel):
	name : str
	depart : str
	field : str

class newRule(BaseModel):
	name : str 
	depart : str
	field : str
	title : str
	sequence : str
	documents : str
	state : str


@app.on_event("startup")
async def setting():
	global new_model, new_processor
	new_model,new_processor = await model()
	if(new_model) : print("모델 세팅완료")
	if(new_processor) : print("프로세서 세팅완료")

	return;

@app.post("/chatStart")
async def question() :
	 
	startmessage = startMessage(new_model, new_processor)
	return {"response" : startmessage}



@app.post("/question")
async def question(request : Message):
	message = request.text
	ragData = searchData(message)
	result = answer(new_model,new_processor,message,ragData)
	return {"response" : result} 

@app.post("/findrule")
async def findRule(request : Rule) :
	print("---!---")
	name = request.name
	depart = request.depart
	field = request.field
	result = searchRules(depart,field,name)

	return {"response" : result}

@app.post("/relatedRule") 
async def relatedRule(request : Rule) : 
	print("----연관 검색중-----")
	sentences = request.name
	result = searchRules("","",sentences)
	
	return {"response" : result}

@app.post("/saveRule")
async def saveRule(request : newRule) :
	print("----저장중----")
	newName = request.name
	newDepart = request.depart
	newField = request.field
	newTitle = request.title
	newDocuments = request.documents
	newSequence = request.sequence
	newState = request.state
	
	metadatas = {"depart" : newDepart, "field" : newField, "title" : newTitle}
	
	# 새로운데이터인경우
	if(newState == "true") :
		ids = newSequence + " "+ newTitle + " " + newName
	# 업데이트인경우
	else :
		ids = newSequence
	print(ids)

	#값이 제대로 안들어갔으면 실패  메세지
	if(newField == "None" or newTitle == "None") : 
		return {"response" : "Not insert value(field or title)"}
	
	result = addData(ids, metadatas, newDocuments)
	
	if(result == False) : 
		return {"response" : "can not insert data in vectorDB"}
	else : 
		return {"response" : "Success"}

@app.post('/deleteRule')
async def deleteRule(request : Rule):
	print("-----삭제중------")
	print(request.name)
	result = deleteData(request.name)
	if(result == False) :
                return {"response" : "can not delete data"}
	else :
		 return {"response" : "Success"}






# uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "server:app",  # 파일 이름과 FastAPI 인스턴스
        host="0.0.0.0",  # 외부에서도 접근 가능
        port=8060,      # 실행할 포트
        reload=True     # 코드 변경 시 자동으로 서버 재시작
    )
