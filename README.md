\# Chatbot Prediksi Bunga



Service chatbot berbasis Flask untuk sistem prediksi penjualan bunga.



Chatbot ini menggunakan LangChain dan Gemini 2.5 Flash untuk menyusun jawaban natural berdasarkan data dari MongoDB Atlas. Service juga memiliki fallback manual agar chatbot tetap dapat menjawab ketika API Gemini terkena limit atau error.



\## Fitur



\- Endpoint health check: `/api/health`

\- Endpoint chatbot AI: `/api/chat`

\- Endpoint fallback manual: `/api/chat/manual`

\- Membaca data dari MongoDB Atlas

\- Menjawab ringkasan dashboard

\- Menampilkan produk dengan prediksi penjualan tertinggi

\- Menampilkan produk dengan stok rendah

\- Menjelaskan model Multiple Linear Regression



\## Teknologi



\- Python

\- Flask

\- LangChain

\- Gemini 2.5 Flash

\- MongoDB Atlas

\- Flask-CORS

\- python-dotenv



\## Environment Variable



Buat file `.env` berdasarkan `.env.example`.



```env

MONGODB\_URI=mongodb+srv://USERNAME:PASSWORD@cluster.example.mongodb.net/?appName=ClusterPrediksiBunga

MONGODB\_DATABASE=prediksi\_bunga

GOOGLE\_API\_KEY=your\_google\_gemini\_api\_key\_here

