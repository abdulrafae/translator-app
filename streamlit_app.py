import streamlit as st
from transformers import MT5ForConditionalGeneration, T5Tokenizer
#from transformers import pipeline

st.title("Stevens Translator")

#Languages = {'afrikaans':'af','albanian':'sq','amharic':'am','arabic':'ar','armenian':'hy','azerbaijani':'az','basque':'eu','belarusian':'be','bengali':'bn','bosnian':'bs','bulgarian':'bg','catalan':'ca','cebuano':'ceb','chichewa':'ny','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','corsican':'co','croatian':'hr','czech':'cs','danish':'da','dutch':'nl','english':'en','esperanto':'eo','estonian':'et','filipino':'tl','finnish':'fi','french':'fr','frisian':'fy','galician':'gl','georgian':'ka','german':'de','greek':'el','gujarati':'gu','haitian creole':'ht','hausa':'ha','hawaiian':'haw','hebrew':'iw','hebrew':'he','hindi':'hi','hmong':'hmn','hungarian':'hu','icelandic':'is','igbo':'ig','indonesian':'id','irish':'ga','italian':'it','japanese':'ja','javanese':'jw','kannada':'kn','kazakh':'kk','khmer':'km','korean':'ko','kurdish (kurmanji)':'ku','kyrgyz':'ky','lao':'lo','latin':'la','latvian':'lv','lithuanian':'lt','luxembourgish':'lb','macedonian':'mk','malagasy':'mg','malay':'ms','malayalam':'ml','maltese':'mt','maori':'mi','marathi':'mr','mongolian':'mn','myanmar (burmese)':'my','nepali':'ne','norwegian':'no','odia':'or','pashto':'ps','persian':'fa','polish':'pl','portuguese':'pt','punjabi':'pa','romanian':'ro','russian':'ru','samoan':'sm','scots gaelic':'gd','serbian':'sr','sesotho':'st','shona':'sn','sindhi':'sd','sinhala':'si','slovak':'sk','slovenian':'sl','somali':'so','spanish':'es','sundanese':'su','swahili':'sw','swedish':'sv','tajik':'tg','tamil':'ta','telugu':'te','thai':'th','turkish':'tr','turkmen':'tk','ukrainian':'uk','urdu':'ur','uyghur':'ug','uzbek':'uz','vietnamese':'vi','welsh':'cy','xhosa':'xh','yiddish':'yi','yoruba':'yo','zulu':'zu'}
Languages = {'arabic':'ar','bengali':'bn','chinese (simplified)':'zh-cn','chinese (traditional)':'zh-tw','czech':'cs','danish':'da','dutch':'nl','english':'en','french':'fr','german':'de','greek':'el','gujarati':'gu','hindi':'hi','japanese':'ja','korean':'ko','urdu':'ur'}

#mname = 'facebook/nllb-200-distilled-600M'
mname = 'google/mt5-xl'
model = MT5ForConditionalGeneration.from_pretrained(mname)
tokenizer = T5Tokenizer.from_pretrained(mname)
#pipe = pipeline("translation_en_to_fr", model=mname)

text = st.text_area("Enter text:",height=None,max_chars=None,key=None,help="Enter your text here")

option1 = st.selectbox('Input language',
                      ('english', 'arabic', 'bengali', 'chinese (simplified)', 'chinese (traditional)', 'czech', 'danish', 'dutch', 'french', 'german', 'hindi', 'japanese', 'korean', 'urdu'))

option2 = st.selectbox('Output language',
                       ('english', 'arabic', 'bengali', 'chinese (simplified)', 'chinese (traditional)', 'czech', 'danish', 'dutch', 'french', 'german', 'hindi', 'japanese', 'korean', 'urdu'))

temp = 1.0
max_length = 128
topk = 50
topp = 1.0
rep_pen = 1.0

value1 = Languages[option1]
value2 = Languages[option2]
task = value1+"-"+value2
if st.button('Translate Sentence'):
    if text == "":
        st.warning('Please **enter text** for translation')

    else:
        line = line = task + ": " + text
        input_ids = tokenizer(line, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids=input_ids, do_sample=True,temperature=temp, max_length=max_length, top_k=topk, top_p=topp, repetition_penalty= rep_pen )
        #out = pipe(line)
        out = tokenizer.decode(output_ids[0])

        # Remove pad and eos tokens.
        out = out.strip().replace('<pad>','').replace('</s>','').replace("<extra_id_0>","").replace("<extra_id_1>","").strip(" ")

        # Fix zero-width joiner issue.
        out = out.replace("\u0dca \u0dbb", "\u0dca\u200d\u0dbb").replace("\u0dca \u0dba", "\u0dca\u200d\u0dba")
        
        #translate = translator.translate(text,lang_src=value1,lang_tgt=value2)
        st.info(str(out))

        st.success("Translation is **successfully** completed!")
        st.balloons()
else:
    pass
