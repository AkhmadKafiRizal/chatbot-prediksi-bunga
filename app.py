from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import traceback

load_dotenv()

app = Flask(__name__)
CORS(app)

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "prediksi_bunga")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"
mongo_client = None


def get_db():
    global mongo_client

    if not MONGODB_URI:
        raise ValueError("MONGODB_URI belum diatur di file .env")

    if mongo_client is None:
        mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000
        )

    return mongo_client[MONGODB_DATABASE]


def get_llm():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY belum diatur di file .env")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "LangChain + Gemini Chatbot API Prediksi Bunga berjalan",
        "service": "langchain-gemini-chatbot-api",
        "model": GEMINI_MODEL
    })


@app.route("/api/health", methods=["GET"])
def health():
    try:
        db = get_db()

        penjualans_count = db["penjualans"].count_documents({})
        products_count = db["products"].count_documents({})
        prediction_results_count = db["prediction_results"].count_documents({})

        google_key_ready = bool(GOOGLE_API_KEY)

        return jsonify({
            "success": True,
            "service": "langchain-gemini-chatbot-api",
            "model": GEMINI_MODEL,
            "database": MONGODB_DATABASE,
            "mongodb_connected": True,
            "gemini_key_ready": google_key_ready,
            "collections": {
                "penjualans": penjualans_count,
                "products": products_count,
                "prediction_results": prediction_results_count
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "service": "langchain-gemini-chatbot-api",
            "mongodb_connected": False,
            "error": str(e)
        }), 500


def get_dashboard_summary():
    db = get_db()

    total_transactions = db["penjualans"].count_documents({})
    total_products = db["products"].count_documents({})
    total_predictions = db["prediction_results"].count_documents({})

    prediction_pipeline = [
        {
            "$group": {
                "_id": None,
                "total_prediction": {"$sum": "$predicted_sales"}
            }
        }
    ]

    prediction_result = list(db["prediction_results"].aggregate(prediction_pipeline))
    total_prediction = prediction_result[0]["total_prediction"] if prediction_result else 0

    latest_prediction = db["prediction_results"].find_one(
        {},
        {"_id": 0, "tanggal": 1},
        sort=[("tanggal", -1)]
    )

    latest_prediction_date = latest_prediction.get("tanggal") if latest_prediction else None

    return {
        "total_transactions": total_transactions,
        "total_products": total_products,
        "total_predictions": total_predictions,
        "total_prediction": total_prediction,
        "latest_prediction_date": latest_prediction_date
    }


def get_low_stock_products():
    db = get_db()

    products = list(db["products"].find(
        {},
        {
            "_id": 1,
            "id": 1,
            "product_id": 1,
            "nama_bunga": 1,
            "name": 1,
            "nama": 1,
            "stok_saat_ini": 1,
            "stock": 1,
            "stok_minimum": 1,
            "minimum_stock": 1,
            "min_stock": 1,
            "satuan": 1
        }
    ))

    cleaned_products = []
    stock_fields_found = False

    for product in products:
        current_stock = get_first_number(product, ["stok_saat_ini", "stock"])
        minimum_stock = get_first_number(product, ["stok_minimum", "minimum_stock", "min_stock"])

        if current_stock is None or minimum_stock is None:
            continue

        stock_fields_found = True

        if current_stock > minimum_stock:
            continue

        cleaned_products.append({
            "product_id": product.get("id") or product.get("product_id") or str(product.get("_id")),
            "nama_bunga": product.get("nama_bunga") or product.get("name") or product.get("nama") or "Tanpa Nama",
            "stok_saat_ini": current_stock,
            "stok_minimum": minimum_stock,
            "satuan": product.get("satuan", "")
        })

    return {
        "products": cleaned_products[:10],
        "total_low_stock": len(cleaned_products),
        "stock_fields_found": stock_fields_found
    }


def get_first_number(data, keys):
    for key in keys:
        if key not in data:
            continue

        value = data.get(key)

        if value is None or value == "":
            continue

        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue

    return None


def get_top_prediction_products():
    db = get_db()

    predictions = list(db["prediction_results"].find(
        {},
        {
            "_id": 0,
            "product_id": 1,
            "predicted_sales": 1,
            "tanggal": 1
        }
    ).sort("predicted_sales", -1).limit(5))

    cleaned_predictions = []

    for prediction in predictions:
        product_id = prediction.get("product_id")

        product = db["products"].find_one(
            {"_id": product_id},
            {
                "_id": 1,
                "nama_bunga": 1
            }
        )

        product_name = product.get("nama_bunga") if product else f"Produk ID {product_id}"

        cleaned_predictions.append({
            "product_id": product_id,
            "nama_bunga": product_name,
            "predicted_sales": prediction.get("predicted_sales", 0),
            "tanggal": prediction.get("tanggal")
        })

    return cleaned_predictions


def get_model_explanation():
    return {
        "algorithm": "Multiple Linear Regression",
        "features": ["harga", "promo", "weekday", "month"],
        "target": "jumlah",
        "training_strategy": "Model dilatih per produk agar prediksi lebih spesifik untuk setiap jenis bunga.",
        "data_source": "MongoDB Atlas database prediksi_bunga"
    }


def detect_intent(message):
    text = message.lower().strip()

    if any(keyword in text for keyword in ["ringkasan", "dashboard", "total", "jumlah data", "summary"]):
        return "dashboard_summary"

    low_stock_keywords = [
        "stok rendah",
        "stoknya rendah",
        "stok yang rendah",
        "stock rendah",
        "stocknya rendah",
        "low stock",
        "stok minimum",
        "minimum stok",
        "restock",
        "stok menipis",
        "stok habis",
        "stok kurang",
        "persediaan rendah"
    ]

    if any(keyword in text for keyword in low_stock_keywords):
        return "low_stock"

    if any(word in text for word in ["stok", "stock", "persediaan"]) and any(word in text for word in ["rendah", "minimum", "menipis", "habis", "kurang", "restock"]):
        return "low_stock"

    if any(keyword in text for keyword in ["prediksi tertinggi", "paling tinggi", "tertinggi", "produk tertinggi"]):
        return "top_prediction"

    if any(keyword in text for keyword in ["model", "algoritma", "multiple linear regression", "linear regression", "fitur", "target", "variabel"]):
        return "model_explanation"

    return "general"


def build_context_by_intent(intent):
    if intent == "dashboard_summary":
        data = get_dashboard_summary()

        manual_answer = (
            f"Ringkasan sistem saat ini: terdapat {data['total_transactions']} data penjualan, "
            f"{data['total_products']} jenis produk bunga, "
            f"{data['total_predictions']} data hasil prediksi, "
            f"dan total prediksi penjualan sebesar {int(data['total_prediction'])}."
        )

        if data.get("latest_prediction_date"):
            manual_answer += f" Periode prediksi terbaru adalah {data['latest_prediction_date']}."

        return data, manual_answer

    if intent == "low_stock":
        data = get_low_stock_products()
        products = data["products"]

        if not data["stock_fields_found"]:
            manual_answer = "Data stok produk belum dapat dibaca saat ini. Pastikan data stok_saat_ini dan stok_minimum tersedia."
        elif not products:
            manual_answer = "Tidak ada produk dengan stok rendah saat ini."
        else:
            items = []

            for index, product in enumerate(products, start=1):
                unit = f" {product['satuan']}" if product.get("satuan") else ""
                items.append(
                    f"{index}. {product['nama_bunga']} - "
                    f"stok {product['stok_saat_ini']}{unit}, minimum {product['stok_minimum']}{unit}"
                )

            manual_answer = "Produk stok rendah teratas:\n" + "\n".join(items)

            if data["total_low_stock"] > len(products):
                manual_answer += f"\nMasih ada {data['total_low_stock'] - len(products)} produk stok rendah lainnya."

        return data, manual_answer

    if intent == "top_prediction":
        data = get_top_prediction_products()

        if not data:
            manual_answer = "Data prediksi belum tersedia."
        else:
            items = []

            for prediction in data:
                items.append(
                    f"{prediction['nama_bunga']} "
                    f"dengan prediksi {int(prediction['predicted_sales'])}"
                )

            manual_answer = "Produk dengan prediksi penjualan tertinggi: " + "; ".join(items)

        return data, manual_answer

    if intent == "model_explanation":
        data = get_model_explanation()

        manual_answer = (
            "Sistem ini menggunakan Multiple Linear Regression untuk memprediksi jumlah penjualan bunga. "
            "Model dilatih per produk dengan fitur harga, promo, weekday, dan month. "
            "Target yang diprediksi adalah jumlah penjualan."
        )

        return data, manual_answer

    data = {
        "capabilities": [
            "Menjawab ringkasan dashboard",
            "Menampilkan produk dengan prediksi tertinggi",
            "Menampilkan produk dengan stok rendah",
            "Menjelaskan model Multiple Linear Regression"
        ]
    }

    manual_answer = (
        "Saya adalah chatbot asisten sistem prediksi penjualan bunga. "
        "Saya bisa membantu menjawab informasi tentang dashboard, stok rendah, hasil prediksi, "
        "dan penjelasan model Multiple Linear Regression."
    )

    return data, manual_answer


def generate_langchain_answer(question, intent, context_data, manual_answer):
    llm = get_llm()

    prompt = f"""
Kamu adalah chatbot asisten untuk sistem prediksi penjualan bunga.

Aturan jawaban:
1. Jawab dalam Bahasa Indonesia.
2. Jawab singkat, jelas, dan profesional.
3. Gunakan hanya data konteks yang diberikan.
4. Jangan mengarang angka, nama produk, atau informasi yang tidak ada.
5. Kalau data benar-benar tidak cukup, katakan bahwa data belum tersedia. Kalau jawaban fallback menyatakan tidak ada produk stok rendah, jangan ubah menjadi data belum tersedia.
6. Jangan menyebut detail teknis internal seperti JSON, MongoDB query, atau Python kecuali user bertanya teknis.
7. Jika konteks berisi daftar produk, sebutkan produk-produk pentingnya secara rapi.

Pertanyaan user:
{question}

Intent:
{intent}

Data konteks:
{context_data}

Jawaban fallback manual:
{manual_answer}

Buat jawaban final yang lebih natural berdasarkan data di atas.
"""

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content.strip()

    return str(response).strip()


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "success": False,
                "answer": "Pertanyaan tidak boleh kosong."
            }), 400

        intent = detect_intent(message)
        context_data, manual_answer = build_context_by_intent(intent)

        if intent == "low_stock":
            return jsonify({
                "success": True,
                "question": message,
                "intent": intent,
                "answer": manual_answer,
                "source": "manual_low_stock"
            })

        answer_source = "gemini"

        try:
            answer = generate_langchain_answer(
                question=message,
                intent=intent,
                context_data=context_data,
                manual_answer=manual_answer
            )

            if not answer:
                answer = manual_answer
                answer_source = "manual_fallback_empty_response"

        except Exception as ai_error:
            answer = manual_answer
            answer_source = "manual_fallback_gemini_error"

            print("Gemini/LangChain error:")
            print(str(ai_error))
            print(traceback.format_exc())

        return jsonify({
            "success": True,
            "question": message,
            "intent": intent,
            "answer": answer,
            "source": answer_source
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/chat/manual", methods=["POST"])
def chat_manual():
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "success": False,
                "answer": "Pertanyaan tidak boleh kosong."
            }), 400

        intent = detect_intent(message)
        context_data, manual_answer = build_context_by_intent(intent)

        return jsonify({
            "success": True,
            "question": message,
            "intent": intent,
            "answer": manual_answer,
            "source": "manual_only"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
