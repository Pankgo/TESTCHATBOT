from transformers import MllamaForConditionalGeneration,MllamaProcessor
import torch
import requests
import time
import re

async def model():
	model = MllamaForConditionalGeneration.from_pretrained(
  	'Bllossom/llama-3.2-Korean-Bllossom-AICA-5B',
  	torch_dtype=torch.bfloat16,
  	device_map='auto'
	)
	processor = MllamaProcessor.from_pretrained('Bllossom/llama-3.2-Korean-Bllossom-AICA-5B')
	
	return(model,processor)

#처음 소개말
def startMessage(model,processor) :
	message = [
		{'role' : 'user',
		'content' : [
		{
			'type' : 'text',
			'text' : 
				f'당신은 학교 규정에 대해 소개하는  챗봇 시스템 "매시" 입니다.'
				f"당신에 대해 친절하게 소개하고 끝말은 항상 존댓말을 사용하도록 하세요"
				f"대상은 학생과 교직원이야"
				f"되도록이면 50글자 이내로 간략하게 설명해줘"
		}]},]
	input_text = processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)

	inputs = processor(
                images=None,
                text=input_text,
                add_special_tokens=False,
                return_tensors="pt",).to(model.device)

	output = model.generate(**inputs,max_new_tokens=512,temperature=0.1,eos_token_id=processor.tokenizer.convert_tokens_to_ids('<|eot_id|>'),use_cache=False)

	result = processor.decode(output[0])

	answer = result.split('<|end_header_id|>')
	answer = re.sub(r"<\|eot_id\|>", "", answer[3])
	
	print(answer)

	return(answer)
	


#프롬프트 생성
def answer(model,processor,question,ragData) : 
	print("====생성중====")
	start = time.time()
	time.sleep(1)
	
	if not ragData :
		messages = [
			{'role':'user',
			'content':[
			{
				'type' :'text',
				'text' :
					f"밑에 있는 문장을 그대로 답변해줘"
					f"참고 문장 : 해당 질문은 답변 할 수 없습니다. 규정과 관련된 질문만 해주세요"
			}]},]
	else :
		messages = [
  			{'role': 'user','content': [
    			{
				'type': 'text',
				'text':
				f"아래에 있는  질문에 대해 아래 참고 데이터를 기반으로  정리해서 답변해주세요." 
                		f"명확하고 간결하게 답변해주세요"       
				f"답변을 할 때는 주어진 참고데이터를 정리해서 말씀해주세요 내용을 그대로 전달하지마세요"
				f"마지막에는 상대방에게 어떤 데이터를 참고해라고 몇조에 해당하는지을 알려줘"
				f"질문: {question}\n\n"
                        	f"참고 데이터:\n{ragData}"}]}, ]

	input_text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

	inputs = processor(
    		images=None,
    		text=input_text,
    		add_special_tokens=False,
    		return_tensors="pt",).to(model.device)

	output = model.generate(**inputs,max_new_tokens=512,temperature=0.1,eos_token_id=processor.tokenizer.convert_tokens_to_ids('<|eot_id|>'),use_cache=False)
	
	result = processor.decode(output[0])

	answer = result.split('<|end_header_id|>')
	answer = re.sub(r"<\|eot_id\|>", "", answer[3])
	
	print(answer)
	print(f"{time.time()-start:.4f} sec")
	
	return(answer)
