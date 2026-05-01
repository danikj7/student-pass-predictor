import gradio as gr
import joblib

# model load
model = joblib.load("model.pkl")

def predict(hours, attendance):
    try:
        prediction = model.predict([[hours, attendance]])[0]

        if prediction == 1:
            return "Pass ✅"
        else:
            return "Fail ❌"
    except:
        return "Invalid input"

# UI
with gr.Blocks() as app:
    gr.Markdown("# 🎓 Student Pass Predictor")

    hours = gr.Number(label="Study Hours")
    attendance = gr.Number(label="Attendance (%)")

    output = gr.Textbox(label="Result")

    btn = gr.Button("Predict")
    btn.click(predict, inputs=[hours, attendance], outputs=output)

app.launch()