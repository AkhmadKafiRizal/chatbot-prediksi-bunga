from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import date, datetime
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

MONTH_NAMES_ID = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember"
}


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


def parse_prediction_date(value):
    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)

    if not value:
        return None

    date_text = str(value).strip()

    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_text[:10], date_format)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(date_text.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_prediction_date(value):
    parsed_date = parse_prediction_date(value)

    if not parsed_date:
        raw_value = str(value) if value else None
        return {
            "date_label": raw_value,
            "period_label": raw_value,
            "iso_date": raw_value
        }

    month_name = MONTH_NAMES_ID.get(parsed_date.month, parsed_date.strftime("%B"))

    return {
        "date_label": f"{parsed_date.day} {month_name} {parsed_date.year}",
        "period_label": f"{month_name} {parsed_date.year}",
        "iso_date": parsed_date.strftime("%Y-%m-%d")
    }


def get_latest_prediction_period():
    db = get_db()

    latest_prediction = db["prediction_results"].find_one(
        {},
        {"_id": 0, "tanggal": 1, "updated_at": 1},
        sort=[("tanggal", -1)]
    )

    if not latest_prediction or not latest_prediction.get("tanggal"):
        return {
            "latest_prediction_date": None,
            "date_label": None,
            "period_label": None,
            "total_predictions_on_date": 0,
            "total_prediction_on_date": 0
        }

    latest_date = latest_prediction.get("tanggal")
    formatted_date = format_prediction_date(latest_date)

    prediction_pipeline = [
        {"$match": {"tanggal": latest_date}},
        {
            "$group": {
                "_id": "$tanggal",
                "total_predictions_on_date": {"$sum": 1},
                "total_prediction_on_date": {"$sum": "$predicted_sales"}
            }
        }
    ]

    prediction_result = list(db["prediction_results"].aggregate(prediction_pipeline))
    prediction_summary = prediction_result[0] if prediction_result else {}

    return {
        "latest_prediction_date": latest_date,
        "date_label": formatted_date["date_label"],
        "period_label": formatted_date["period_label"],
        "iso_date": formatted_date["iso_date"],
        "total_predictions_on_date": prediction_summary.get("total_predictions_on_date", 0),
        "total_prediction_on_date": prediction_summary.get("total_prediction_on_date", 0),
        "last_updated_at": latest_prediction.get("updated_at")
    }


def get_latest_sale_period():
    db = get_db()

    latest_sale = db["penjualans"].find_one(
        {},
        {"_id": 0, "tanggal": 1},
        sort=[("tanggal", -1)]
    )

    if not latest_sale or not latest_sale.get("tanggal"):
        return {
            "latest_sale_date": None,
            "sale_date_label": None,
            "sale_period_label": None
        }

    latest_date = latest_sale.get("tanggal")
    formatted_date = format_prediction_date(latest_date)

    return {
        "latest_sale_date": latest_date,
        "sale_date_label": formatted_date["date_label"],
        "sale_period_label": formatted_date["period_label"],
        "sale_iso_date": formatted_date["iso_date"]
    }


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
        "developer": "Mas Kafi dan Tim",
        "training_strategy": "Model dilatih per produk agar prediksi lebih spesifik untuk setiap jenis bunga.",
        "evaluation_metrics": ["MAE", "RMSE", "R2"],
        "data_source": "MongoDB Atlas database prediksi_bunga",
        "main_reason": (
            "Aplikasi Web FloraPredict memprediksi jumlah penjualan dalam bentuk angka. "
            "Penjualan dipengaruhi beberapa fitur input seperti harga, promo, weekday, dan month. "
            "Multiple Linear Regression sesuai karena dapat memodelkan hubungan beberapa variabel input "
            "terhadap satu target numerik."
        ),
        "simple_linear_reason": (
            "Linear Regression biasa umumnya hanya memakai satu variabel input, sedangkan data sistem ini "
            "menggunakan beberapa variabel input seperti harga, promo, weekday, dan month."
        ),
        "complex_model_reason": (
            "Fokus proyek adalah membuat sistem prediksi yang jelas alurnya, mudah dijelaskan, dan sesuai "
            "dengan data yang tersedia. Multiple Linear Regression cukup untuk baseline prediksi, mudah "
            "dievaluasi, dan stabil diintegrasikan ke Laravel, Flask, MongoDB, dan Flutter."
        )
    }


def detect_intent(message):
    text = message.lower().strip()

    creator_keywords = [
        "siapa pencipta",
        "siapa pembuat",
        "siapa yang membuat",
        "dibuat oleh siapa",
        "pembuat ai",
        "pencipta ai",
        "creator ai",
        "siapa developer",
        "developer ai",
        "siapa pengembang",
        "pengembang ai",
        "author ai"
    ]

    if any(keyword in text for keyword in creator_keywords) and any(keyword in text for keyword in ["flora", "florapredict", "flora predict", "ai"]):
        return "creator_info"

    reason_keywords = ["kenapa", "mengapa", "kok", "padahal"]
    latest_reference_keywords = [
        "data terakhir",
        "terakhirnya",
        "31 januari",
        "januari 2024",
        "tanggal sekarang",
        "sekarang",
        "hari ini",
        "tahun 2026",
        "bulan 5",
        "mei 2026"
    ]

    if any(keyword in text for keyword in reason_keywords) and any(keyword in text for keyword in latest_reference_keywords):
        return "latest_prediction_reason"

    latest_prediction_keywords = [
        "tanggal prediksi terakhir",
        "prediksi terakhir tanggal",
        "prediksi terbaru tanggal",
        "data terakhir yang diprediksi",
        "data terakhir diprediksi",
        "terakhir yang diprediksi",
        "terakhir diprediksi",
        "hasil prediksi terakhir",
        "hasil prediksi terbaru",
        "periode prediksi terakhir",
        "periode prediksi terbaru",
        "prediksi terakhir",
        "prediksi terbaru",
        "last prediction"
    ]

    if any(keyword in text for keyword in latest_prediction_keywords):
        return "latest_prediction_date"

    if any(keyword in text for keyword in ["prediksi", "diprediksi", "hasil prediksi"]) and any(keyword in text for keyword in ["terakhir", "terbaru", "tanggal", "bulan", "tahun", "periode"]):
        return "latest_prediction_date"

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

    if any(keyword in text for keyword in ["linear regression biasa", "linear regression sederhana", "simple linear regression"]) or (
        "linear regression" in text and any(keyword in text for keyword in ["bukan", "kenapa bukan", "mengapa bukan", "beda", "bedanya", "biasa"])
    ):
        return "why_not_simple_linear_regression"

    if any(keyword in text for keyword in ["model yang lebih kompleks", "model lebih kompleks", "model kompleks", "model yang lebih canggih", "model lebih canggih"]) or (
        any(keyword in text for keyword in ["random forest", "svm", "xgboost", "lstm", "neural network", "deep learning"]) and any(keyword in text for keyword in ["kenapa", "mengapa", "pakai", "tidak"])
    ):
        return "why_not_complex_model"

    if any(keyword in text for keyword in ["model", "metode", "algoritma", "multiple linear regression", "linear regression", "fitur", "target", "variabel", "mae", "rmse", "r2", "r²"]):
        return "model_explanation"

    return "general"


def build_context_by_intent(intent):
    if intent == "creator_info":
        data = {
            "creator_name": "Akhmad Kafi Rizal",
            "creator_nickname": "Mas Kahfi",
            "project": "AI FloraPredict"
        }

        manual_answer = "Pencipta AI FloraPredict ini adalah Akhmad Kafi Rizal, yang biasa dipanggil Mas Kahfi."

        return data, manual_answer

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

    if intent == "latest_prediction_date":
        data = get_latest_prediction_period()

        if not data.get("latest_prediction_date"):
            manual_answer = "Data prediksi terbaru belum tersedia. Jalankan generate prediksi terlebih dahulu."
        else:
            manual_answer = (
                f"Data prediksi terakhir tersedia untuk tanggal {data['date_label']} "
                f"(periode {data['period_label']})."
            )

            if data.get("total_predictions_on_date"):
                manual_answer += (
                    f" Pada periode tersebut terdapat {data['total_predictions_on_date']} data produk "
                    f"dengan total prediksi {int(data.get('total_prediction_on_date', 0))} tangkai."
                )

        return data, manual_answer

    if intent == "latest_prediction_reason":
        data = get_latest_prediction_period()
        sale_data = get_latest_sale_period()
        data.update(sale_data)

        if not data.get("latest_prediction_date"):
            manual_answer = (
                "Karena hasil prediksi belum tersedia di database. "
                "Tanggal hari ini tidak otomatis membuat data prediksi baru. "
                "Jalankan generate prediksi terlebih dahulu agar periode prediksi tersimpan."
            )
        else:
            manual_answer = (
                f"Karena yang saya baca adalah hasil prediksi yang sudah tersimpan, bukan tanggal hari ini. "
                f"Hasil prediksi terbaru yang tersimpan masih tanggal {data['date_label']} "
                f"(periode {data['period_label']}). "
                "Tanggal sekarang tidak otomatis membuat prediksi baru."
            )

            if data.get("sale_date_label"):
                manual_answer += (
                    f" Data penjualan terakhir tercatat sampai {data['sale_date_label']}, "
                    "tetapi hasil prediksi terbaru tetap mengikuti data yang terakhir digenerate."
                )

            manual_answer += (
                " Kalau ingin periode 2026 muncul sebagai prediksi terbaru, jalankan Generate Prediksi "
                "untuk periode tersebut terlebih dahulu."
            )

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
            "Aplikasi Web FloraPredict yang dikembangkan oleh Mas Kafi dan Tim menggunakan Multiple Linear Regression. "
            "Alasannya, sistem ini memprediksi jumlah penjualan dalam bentuk angka, dan penjualan dipengaruhi beberapa fitur "
            "seperti harga, promo, weekday, dan month. Multiple Linear Regression cocok karena bisa memodelkan hubungan "
            "beberapa variabel input terhadap satu target numerik. Model ini juga mudah dijelaskan, dievaluasi dengan "
            "MAE/RMSE/R², dan di sistem ini dilatih per produk agar hasil prediksinya lebih spesifik."
        )

        return data, manual_answer

    if intent == "why_not_simple_linear_regression":
        data = get_model_explanation()

        manual_answer = (
            "Karena Linear Regression biasa umumnya hanya memakai satu variabel input, sedangkan data FloraPredict "
            "memakai beberapa variabel input, yaitu harga, promo, weekday, dan month. Jadi metode yang lebih sesuai "
            "adalah Multiple Linear Regression."
        )

        return data, manual_answer

    if intent == "why_not_complex_model":
        data = get_model_explanation()

        manual_answer = (
            "Karena fokus proyek yang dikembangkan oleh Mas Kafi dan Tim adalah membuat sistem prediksi yang jelas alurnya, "
            "mudah dijelaskan, dan sesuai dengan data yang tersedia. Multiple Linear Regression sudah cukup untuk baseline "
            "prediksi, mudah dievaluasi, dan hasilnya bisa diintegrasikan ke Laravel, Flask, MongoDB, dan Flutter dengan stabil."
        )

        return data, manual_answer

    data = {
        "capabilities": [
            "Menjawab ringkasan dashboard",
            "Menampilkan produk dengan prediksi tertinggi",
            "Menjawab tanggal dan periode prediksi terbaru",
            "Menampilkan produk dengan stok rendah",
            "Menjelaskan alasan penggunaan Multiple Linear Regression",
            "Menjawab informasi pencipta AI FloraPredict"
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

        if intent == "creator_info":
            return jsonify({
                "success": True,
                "question": message,
                "intent": intent,
                "answer": manual_answer,
                "source": "manual_creator_info"
            })

        if intent == "latest_prediction_date":
            return jsonify({
                "success": True,
                "question": message,
                "intent": intent,
                "answer": manual_answer,
                "source": "manual_latest_prediction_date"
            })

        if intent == "latest_prediction_reason":
            return jsonify({
                "success": True,
                "question": message,
                "intent": intent,
                "answer": manual_answer,
                "source": "manual_latest_prediction_reason"
            })

        if intent in ["model_explanation", "why_not_simple_linear_regression", "why_not_complex_model"]:
            return jsonify({
                "success": True,
                "question": message,
                "intent": intent,
                "answer": manual_answer,
                "source": "manual_model_explanation"
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
