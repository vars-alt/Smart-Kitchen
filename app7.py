import streamlit as st
import numpy as np
import requests
import torch
import faiss
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
import cv2
import base64
import os
from gtts import gTTS
import urllib.parse

# ------------------ Setup ------------------

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="Smart Kitchen 🍳", layout="centered")
st.title("🥕 Smart Kitchen")
st.write("Upload an image of your fridge and get recipes!")

# ------------------ Load Models ------------------

model = load_model("C:\\Users\\medan\\OneDrive\\Documents\\NITK\\AI\\Team 7 - Smart Kitchen\mymodel.h5")

class_labels = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", "carrot", 
    "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant", "garlic", 
    "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce", "mango", "onion", 
    "orange", "paprika", "peas", "pear", "pineapple", "pomegranate", "potato", 
    "raddish", "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", 
    "turnip", "watermelon"
]

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

API_KEY = "26bd601459114d41aadbc72c03d92127"

# ------------------ Functions ------------------

def predict_ingredients(image):
    img_rgb = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)[0]
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_labels = [class_labels[i] for i in top_indices]
    return top_labels

def get_recipes_from_spoonacular(ingredients, number=5):
    ing_str = ",".join(ingredients)
    url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={ing_str}&number={number}&apiKey={API_KEY}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else []

def get_nutrition(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/nutritionWidget.json?apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}

def get_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = bert_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

def generate_instructions_from_ingredients(ingredients):
    text = f"Generate cooking instructions for: {', '.join(ingredients)}.\n"
    input_ids = gpt2_tokenizer.encode(text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)
    output = gpt2_model.generate(input_ids, max_length=150, temperature=0.7, pad_token_id=gpt2_tokenizer.eos_token_id, attention_mask=attention_mask)
    result = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split("instructions:")[-1].strip()

def get_documents_from_recipes(recipes):
    docs = []
    for recipe in recipes:
        title = recipe['title']
        used_ingredients = [i['name'] for i in recipe['usedIngredients']]
        instructions = recipe.get('instructions', '') or generate_instructions_from_ingredients(used_ingredients)
        recipe_id = recipe['id']
        docs.append({
            "title": title,
            "ingredients": used_ingredients,
            "instructions": instructions,
            "id": recipe_id
        })
    return docs

def generate_embeddings(docs):
    return [get_embedding(doc['instructions']) for doc in docs]

def create_faiss_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def retrieve(query, index, docs, k=3):
    query_emb = get_embedding(query)
    D, I = index.search(np.array([query_emb]), k)
    return [docs[i] for i in I[0]]

def generate_recipe_from_retrieved_docs(query, docs):
    context = " ".join([f"Title: {doc['title']}, Ingredients: {', '.join(doc['ingredients'])}, Instructions: {doc['instructions']}" for doc in docs])
    input_text = f"Given the following ingredients and instructions: {context}, generate a new recipe based on the query: {query}"
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape)
    output = gpt2_model.generate(input_ids, max_length=1000, temperature=0.7, pad_token_id=gpt2_tokenizer.eos_token_id, attention_mask=attention_mask)
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

def speak_text_gtts(text, index):
    tts = gTTS(text=text, lang='en')
    audio_path = f"recipe_audio_{index}.mp3"
    tts.save(audio_path)
    with open(audio_path, "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")

# ------------------ Shelf Life Dictionary ------------------

shelf_life = {
    "apple": "4–6 weeks", "banana": "5–7 days", "beetroot": "2–3 weeks", "bell pepper": "1–2 weeks",
    "cabbage": "1–2 months", "capsicum": "1–2 weeks", "carrot": "3–4 weeks", "cauliflower": "1–2 weeks",
    "chilli pepper": "1–2 weeks", "corn": "1–3 days", "cucumber": "1 week", "eggplant": "4–5 days",
    "garlic": "3–5 months", "ginger": "1 month", "grapes": "1–2 weeks", "jalepeno": "1 week",
    "kiwi": "1–4 weeks", "lemon": "3–4 weeks", "lettuce": "7–10 days", "mango": "5–7 days",
    "onion": "1–2 months", "orange": "3–4 weeks", "paprika": "1–2 weeks", "peas": "5–7 days",
    "pear": "1–2 weeks", "pineapple": "3–5 days", "pomegranate": "2 months", "potato": "2–3 months",
    "raddish": "2 weeks", "soy beans": "3–4 days", "spinach": "5–7 days", "sweetcorn": "1–3 days",
    "sweetpotato": "2–3 months", "tomato": "1 week", "turnip": "2–3 weeks", "watermelon": "7–10 days"
}

# ------------------ UI ------------------

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        labels = predict_ingredients(image)

    st.subheader("📦 Average Shelf Life in a Fridge:")
    for l in labels:
        life = shelf_life.get(l, "N/A")
        st.write(f"🕒 {l}: {life}")

    user_query = st.text_input("💬 What kind of dish do you want? (e.g., 'Make a spicy recipe', 'A vegan lunch')", value="How to cook with these ingredients?")

    if st.button("🍽 Generate Recipes with These Ingredients"):
        with st.spinner("Cooking up something tasty with AI..."):
            recipes = get_recipes_from_spoonacular(labels)
            if not recipes:
                st.error("No recipes found!")
            else:
                docs = get_documents_from_recipes(recipes)
                embeddings = generate_embeddings(docs)
                index = create_faiss_index(embeddings)
                retrieved = retrieve(user_query, index, docs)
                ai_recipe = generate_recipe_from_retrieved_docs(user_query, retrieved)

                st.subheader("🔍AI-Generated Recipes:")
                for i, doc in enumerate(retrieved):
                    st.markdown(f"**🍲 {doc['title']}**")
                    st.markdown("**Ingredients:**")
                    st.write(", ".join(doc['ingredients']))
                    st.markdown("**Instructions:**")
                    st.write(doc['instructions'])

                    nutrition = get_nutrition(doc['id'])
                    if nutrition:
                        st.markdown("**🧪 Nutritional Info (per serving):**")
                        st.write(f"Calories: {nutrition.get('calories', 'N/A')}")
                        st.write(f"Protein: {nutrition.get('protein', 'N/A')}")
                        st.write(f"Fat: {nutrition.get('fat', 'N/A')}")
                        st.write(f"Carbohydrates: {nutrition.get('carbs', 'N/A')}")

                    # Add YouTube search link for recipe video
                    recipe_title = doc['title']
                    encoded_title = urllib.parse.quote_plus(recipe_title)
                    search_link = f"https://www.youtube.com/results?search_query={encoded_title}+recipe+video"
                    st.markdown(f"**🎥 Watch the recipe video**: [Click here]({search_link})")

                    speak_text_gtts(f"{doc['title']}. Ingredients: {', '.join(doc['ingredients'])}. Instructions: {doc['instructions']}", i)

                    st.markdown("---")

                # st.subheader("🧠 AI-Created Custom Recipe:")
                # st.write(ai_recipe)
                speak_text_gtts(ai_recipe, "final")

# ------------------ Avatar ------------------

def add_avatar():
    file_path = "avatar.gif"
    with open(file_path, "rb") as file_:
        contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .avatar-container {{
            position: fixed;
            top: 100px;
            left: -160px;
            width: 600px;
            height: 200px;
            z-index: 999;
            animation: floatAround 8s ease-in-out infinite alternate;
        }}

        @keyframes floatAround {{
            0% {{ transform: translate(0, 0); }}
            25% {{ transform: translate(20px, -30px); }}
            50% {{ transform: translate(0, -60px); }}
            75% {{ transform: translate(-20px, -30px); }}
            100% {{ transform: translate(0, 0); }}
        }}
        </style>

        <div class="avatar-container">
            <img src="data:image/gif;base64,{data_url}" style="height: 280px; width: 600px;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

add_avatar()
