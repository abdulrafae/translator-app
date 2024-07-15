import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
#from transformers import pipeline

st.title("Stevens Translator")

#Languages = {'afrikaans':'af','albanian':'sq','amharic':'am','arabic':'ar','armenian':'hy','azerbaijani':'az','basque':'eu','belarusian':'be','bengali':'bn','bosnian':'bs','bulgarian':'bg','catalan':'ca','cebuano':'ceb','chichewa':'ny','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','corsican':'co','croatian':'hr','czech':'cs','danish':'da','dutch':'nl','english':'en','esperanto':'eo','estonian':'et','filipino':'tl','finnish':'fi','french':'fr','frisian':'fy','galician':'gl','georgian':'ka','german':'de','greek':'el','gujarati':'gu','haitian creole':'ht','hausa':'ha','hawaiian':'haw','hebrew':'iw','hebrew':'he','hindi':'hi','hmong':'hmn','hungarian':'hu','icelandic':'is','igbo':'ig','indonesian':'id','irish':'ga','italian':'it','japanese':'ja','javanese':'jw','kannada':'kn','kazakh':'kk','khmer':'km','korean':'ko','kurdish (kurmanji)':'ku','kyrgyz':'ky','lao':'lo','latin':'la','latvian':'lv','lithuanian':'lt','luxembourgish':'lb','macedonian':'mk','malagasy':'mg','malay':'ms','malayalam':'ml','maltese':'mt','maori':'mi','marathi':'mr','mongolian':'mn','myanmar (burmese)':'my','nepali':'ne','norwegian':'no','odia':'or','pashto':'ps','persian':'fa','polish':'pl','portuguese':'pt','punjabi':'pa','romanian':'ro','russian':'ru','samoan':'sm','scots gaelic':'gd','serbian':'sr','sesotho':'st','shona':'sn','sindhi':'sd','sinhala':'si','slovak':'sk','slovenian':'sl','somali':'so','spanish':'es','sundanese':'su','swahili':'sw','swedish':'sv','tajik':'tg','tamil':'ta','telugu':'te','thai':'th','turkish':'tr','turkmen':'tk','ukrainian':'uk','urdu':'ur','uyghur':'ug','uzbek':'uz','vietnamese':'vi','welsh':'cy','xhosa':'xh','yiddish':'yi','yoruba':'yo','zulu':'zu'}
Languages = {'arabic':'ar','bengali':'bn','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','czech':'cs','danish':'da','dutch':'nl','english':'en','french':'fr','german':'de','greek':'el','gujarati':'gu','hindi':'hi','japanese':'ja','korean':'ko','urdu':'ur'}


mname = 'jbochi/madlad400-3b-mt'
#mname = 'facebook/nllb-200-distilled-600M'
mname = 'google-t5/t5-large'
model = T5ForConditionalGeneration.from_pretrained(mname)
tokenizer = T5Tokenizer.from_pretrained(mname)
#pipe = pipeline("translation_en_to_fr", model=mname)

text = st.text_area("Enter text:",height=None,max_chars=None,key=None,help="Enter your text here")

#option1 = st.selectbox('Input language',
#                      ('english', 'arabic', 'bengali', 'chinese (simplified)', 'chinese (traditional)', 'czech', 'danish', 'dutch', 'french', 'german', 'hindi', 'japanese', 'korean', 'urdu'))

#option2 = st.selectbox('Output language',
#                       ('english', 'arabic', 'bengali', 'chinese (simplified)', 'chinese (traditional)', 'czech', 'danish', 'dutch', 'french', 'german', 'hindi', 'japanese', 'korean', 'urdu'))
#option1 = st.selectbox('Input language',
#                      ('english', 'afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'azerbaijani', 'basque', 'belarusian', 'bengali', 'bosnian', 'bulgarian', 'catalan', 'cebuano', 'chichewa', 'chinese (simplified)', 'chinese (traditional)', 'corsican', 'croatian', 'czech', 'danish', 'dutch',  'esperanto', 'estonian', 'filipino', 'finnish', 'french', 'frisian', 'galician', 'georgian', 'german', 'greek', 'gujarati', 'haitian creole', 'hausa', 'hawaiian', 'hebrew', 'hindi', 'hmong', 'hungarian', 'icelandic', 'igbo', 'indonesian', 'irish', 'italian', 'japanese', 'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz', 'lao', 'latin', 'latvian', 'lithuanian', 'luxembourgish', 'macedonian', 'malagasy', 'malay', 'malayalam', 'maltese', 'maori', 'marathi', 'mongolian', 'myanmar (burmese)', 'nepali', 'norwegian', 'odia', 'pashto', 'persian', 'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'samoan', 'scots gaelic', 'serbian', 'sesotho', 'shona', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese', 'swahili', 'swedish', 'tajik', 'tamil', 'telugu', 'thai', 'turkish', 'turkmen', 'ukrainian', 'urdu', 'uyghur', 'uzbek', 'vietnamese', 'welsh', 'xhosa', 'yiddish', 'yoruba', 'zulu'))

option1 = st.selectbox('Input language', ('english',))

option2 = st.selectbox('Output language',
                       ('french','german'))

temp = 1.0
max_length = 128
topk = 50
topp = 1.0
rep_pen = 1.0

#value1 = Languages[option1]
#value2 = Languages[option2]
#task = value1+"-"+value2
#task = "<"+value1+"2"+value2+">"
task = "translate "+option1.title()+" to "+option2.title()
if st.button('Translate Sentence'):
    if text == "":
        st.warning('Please **enter text** for translation')

    else:
        line = line = task + ": " + str(text)
        #line = line = task + " " + text
        print(line)
        final_result = ''
        with st.spinner('Wait for it...'):
            while final_result == '':
                input_ids = tokenizer(text, return_tensors="pt").input_ids
                #output_ids = model.generate(input_ids=input_ids, do_sample=True,temperature=temp, max_length=max_length, top_k=topk, top_p=topp, repetition_penalty= rep_pen )
                output_ids = model.generate(input_ids=input_ids)
        
                out = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
                # Remove pad and eos tokens.
                out = out.strip().replace('<pad>','').replace('</s>','').replace("<extra_id_0>","").replace("<extra_id_1>","").strip(" ")
        
                # Fix zero-width joiner issue.
                final_result = out.replace("\u0dca \u0dbb", "\u0dca\u200d\u0dbb").replace("\u0dca \u0dba", "\u0dca\u200d\u0dba")
        
        print(final_result)
        st.info(str(final_result))

        st.success("Translation is **successfully** completed!")
        #st.balloons()
else:
    pass
