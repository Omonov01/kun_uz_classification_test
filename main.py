import streamlit as st
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TextClassificationPipeline
from transformers import pipeline

load_model = AutoModelForSequenceClassification.from_pretrained("Kun_uz_classification")
 
load_tokenizer = AutoTokenizer.from_pretrained("Kun_uz_classification")
st.write("Airi.uz jamoasi amaliyotchilari tomonidan tayyorlangan text classification uchun mo'ljallangan model")
st.write("Ishlatish uchun pastdagi maydonga matn kiriting va model sizga kiritilgan matnni qaysi sohaga aloqador ekanligini ko'rsatadi")
input = st.text_area(label='input_areaf',placeholder='matnni shu yerga kiriting',height=350,max_chars = 5000)
try:
      if st.button(label='bashorat qilish'):
            my_pipeline  = pipeline("text-classification", model=load_model, tokenizer=load_tokenizer)
            data = input
            st.info(my_pipeline(data))
except RuntimeError:
      st.info("Iltimos kamroq malumot kiriting")