# Import libraries

import flet as ft
import pandas as pd
import joblib
import os

# Load model

model = joblib.load("models/decision_tree_model.pkl")
model_info = joblib.load("models/model_info.pkl")

# App settings

def main(page: ft.Page):

    page.title = "MediCore AI - Heart Disease Prediction"
    page.bgcolor = "#080E14"
    page.padding = 0
    page.window.width = 1250
    page.window.height = 850
    page.theme_mode = ft.ThemeMode.DARK

    green = "#00E5A8"
    cyan = "#00CFFF"
    card_color = "#111C27"
    text_color = "#F1F5F9"
    muted = "#8B9BAA"

    # Input fields

    age = ft.TextField(
        label="Age",
        hint_text="Enter age"
    )

    sex = ft.Dropdown(
        label="Sex",
        options=[
            ft.DropdownOption(key="0", text="Female"),
            ft.DropdownOption(key="1", text="Male")
        ]
    )

    cp = ft.Dropdown(
        label="Chest Pain Type",
        options=[
            ft.DropdownOption(key="1", text="Typical Angina"),
            ft.DropdownOption(key="2", text="Atypical Angina"),
            ft.DropdownOption(key="3", text="Non-anginal Pain"),
            ft.DropdownOption(key="4", text="Asymptomatic")
        ]
    )

    trestbps = ft.TextField(
        label="Resting Blood Pressure",
        hint_text="Example: 120"
    )

    chol = ft.TextField(
        label="Cholesterol",
        hint_text="Example: 200"
    )

    fbs = ft.Dropdown(
        label="Fasting Blood Sugar",
        options=[
            ft.DropdownOption(key="0", text="False"),
            ft.DropdownOption(key="1", text="True")
        ]
    )

    restecg = ft.Dropdown(
        label="Resting ECG",
        options=[
            ft.DropdownOption(key="0", text="Normal"),
            ft.DropdownOption(key="1", text="ST-T Abnormality"),
            ft.DropdownOption(key="2", text="LV Hypertrophy")
        ]
    )

    thalach = ft.TextField(
        label="Maximum Heart Rate",
        hint_text="Example: 150"
    )

    exang = ft.Dropdown(
        label="Exercise Induced Angina",
        options=[
            ft.DropdownOption(key="0", text="No"),
            ft.DropdownOption(key="1", text="Yes")
        ]
    )

    oldpeak = ft.TextField(
        label="ST Depression",
        hint_text="Example: 1.5"
    )

    slope = ft.Dropdown(
        label="Slope",
        options=[
            ft.DropdownOption(key="1", text="Upsloping"),
            ft.DropdownOption(key="2", text="Flat"),
            ft.DropdownOption(key="3", text="Downsloping")
        ]
    )

    ca = ft.Dropdown(
        label="Number of Major Vessels",
        options=[
            ft.DropdownOption(key="0", text="0"),
            ft.DropdownOption(key="1", text="1"),
            ft.DropdownOption(key="2", text="2"),
            ft.DropdownOption(key="3", text="3")
        ]
    )

    thal = ft.Dropdown(
        label="Thalassemia",
        options=[
            ft.DropdownOption(key="3", text="Normal"),
            ft.DropdownOption(key="6", text="Fixed Defect"),
            ft.DropdownOption(key="7", text="Reversible Defect")
        ]
    )

    result_area = ft.Column(spacing=10)

    # UI components

    def make_card(title, value, icon, color):

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(icon, color=color, size=25),
                    ft.Text(title, color=muted, size=13)
                ]),
                ft.Text(
                    value,
                    size=27,
                    weight=ft.FontWeight.BOLD,
                    color=text_color
                )
            ], spacing=12),
            bgcolor=card_color,
            padding=20,
            border_radius=15,
            expand=True
        )

    def section_title(title, subtitle):

        return ft.Column([
            ft.Text(
                title,
                size=25,
                weight=ft.FontWeight.BOLD,
                color=text_color
            ),
            ft.Text(
                subtitle,
                color=muted,
                size=13
            )
        ], spacing=5)

    def show_message(message):

        page.show_dialog(
            ft.SnackBar(
                content=ft.Text(message)
            )
        )

    # Prediction function

    def predict_disease(e):

        try:

            required_fields = [
                age, sex, cp, trestbps, chol,
                fbs, restecg, thalach, exang,
                oldpeak, slope, ca, thal
            ]

            for field in required_fields:

                if field.value is None or field.value == "":
                    show_message("Please fill in all fields.")
                    return

            patient_data = pd.DataFrame([{
                "age": float(age.value),
                "sex": int(sex.value),
                "cp": int(cp.value),
                "trestbps": float(trestbps.value),
                "chol": float(chol.value),
                "fbs": int(fbs.value),
                "restecg": int(restecg.value),
                "thalach": float(thalach.value),
                "exang": int(exang.value),
                "oldpeak": float(oldpeak.value),
                "slope": int(slope.value),
                "ca": int(ca.value),
                "thal": int(thal.value)
            }])

            prediction = model.predict(patient_data)[0]

            probability = model.predict_proba(patient_data)[0]

            confidence = probability[int(prediction)] * 100

            if prediction == 1:

                result_color = "#FF5364"
                result_text = "Heart Disease Predicted"

                result_description = (
                    "The model classified this input "
                    "in the heart disease class."
                )

            else:

                result_color = green
                result_text = "No Heart Disease Predicted"

                result_description = (
                    "The model classified this input "
                    "in the no heart disease class."
                )

            result_area.controls.clear()

            result_area.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(
                            ft.Icons.FAVORITE,
                            size=45,
                            color=result_color
                        ),

                        ft.Text(
                            result_text,
                            size=23,
                            weight=ft.FontWeight.BOLD,
                            color=result_color
                        ),

                        ft.Text(
                            result_description,
                            color=text_color
                        ),

                        ft.Text(
                            f"Model confidence: {confidence:.2f}%",
                            color=cyan,
                            size=17
                        ),

                        ft.Text(
                            "This is an educational model prediction, "
                            "not a medical diagnosis.",
                            color=muted,
                            size=12
                        )

                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=15),

                    bgcolor=card_color,
                    padding=25,
                    border_radius=15
                )
            )

            page.update()

        except ValueError:

            show_message("Please enter valid numeric values.")

        except Exception as error:

            show_message(f"Prediction error: {error}")

    # Dashboard page

    def dashboard_page():

        return ft.Column([

            section_title(
                "Welcome to MediCore AI",
                "Heart Disease Prediction System"
            ),

            ft.Container(height=15),

            ft.Row([

                make_card(
                    "Accuracy",
                    f"{model_info['accuracy'] * 100:.2f}%",
                    ft.Icons.STAR,
                    green
                ),

                make_card(
                    "Precision",
                    f"{model_info['precision'] * 100:.2f}%",
                    ft.Icons.CHECK_CIRCLE,
                    cyan
                ),

                make_card(
                    "Recall",
                    f"{model_info['recall'] * 100:.2f}%",
                    ft.Icons.FAVORITE,
                    green
                ),

                make_card(
                    "F1 Score",
                    f"{model_info['f1_score'] * 100:.2f}%",
                    ft.Icons.ANALYTICS,
                    cyan
                )

            ], spacing=12),

            ft.Container(height=20),

            ft.Container(
                content=ft.Column([

                    ft.Text(
                        "Decision Tree Model",
                        size=22,
                        weight=ft.FontWeight.BOLD,
                        color=green
                    ),

                    ft.Text(
                        "Machine Learning Classification",
                        color=muted
                    ),

                    ft.Divider(color="#263746"),

                    ft.Text(
                        "This application uses a Decision Tree "
                        "Classifier to predict heart disease "
                        "from patient health information.",
                        color=text_color,
                        size=15
                    ),

                    ft.Text(
                        "Dataset: UCI Heart Disease",
                        color=cyan
                    ),

                    ft.Text(
                        "Algorithm: Decision Tree Classifier",
                        color=cyan
                    ),

                    ft.Text(
                        "Classification: Binary Classification",
                        color=cyan
                    )

                ], spacing=12),

                bgcolor=card_color,
                padding=25,
                border_radius=15
            ),

            ft.Container(height=15),

            ft.Text(
                "Model Training Accuracy: "
                f"{model_info['train_accuracy'] * 100:.2f}%",
                color=muted
            ),

            ft.Text(
                "Model Testing Accuracy: "
                f"{model_info['test_accuracy'] * 100:.2f}%",
                color=muted
            )

        ], spacing=15, scroll=ft.ScrollMode.AUTO)

    # Prediction page

    def prediction_page():

        return ft.Column([

            section_title(
                "Heart Disease Prediction",
                "Enter patient information to generate a prediction"
            ),

            ft.Container(height=10),

            ft.Container(
                content=ft.Column([

                    ft.Text(
                        "Patient Information",
                        size=20,
                        color=green,
                        weight=ft.FontWeight.BOLD
                    ),

                    ft.ResponsiveRow([

                        ft.Container(age, col={"xs": 12, "md": 6}),
                        ft.Container(sex, col={"xs": 12, "md": 6}),
                        ft.Container(cp, col={"xs": 12, "md": 6}),
                        ft.Container(trestbps, col={"xs": 12, "md": 6}),
                        ft.Container(chol, col={"xs": 12, "md": 6}),
                        ft.Container(fbs, col={"xs": 12, "md": 6}),
                        ft.Container(restecg, col={"xs": 12, "md": 6}),
                        ft.Container(thalach, col={"xs": 12, "md": 6}),
                        ft.Container(exang, col={"xs": 12, "md": 6}),
                        ft.Container(oldpeak, col={"xs": 12, "md": 6}),
                        ft.Container(slope, col={"xs": 12, "md": 6}),
                        ft.Container(ca, col={"xs": 12, "md": 6}),
                        ft.Container(thal, col={"xs": 12, "md": 6})

                    ], spacing=12),

                    ft.Container(height=10),

                    ft.Button(
                        content="Predict Heart Disease",
                        icon=ft.Icons.FAVORITE,
                        bgcolor=green,
                        color="#08131A",
                        on_click=predict_disease,
                        height=50
                    )

                ], spacing=15),

                bgcolor=card_color,
                padding=25,
                border_radius=15
            ),

            ft.Container(height=15),

            result_area

        ], spacing=15, scroll=ft.ScrollMode.AUTO)

    # Model analysis page

    def analysis_page():

        image_files = [

            ("Target Distribution", "target_distribution.png"),
            ("Correlation Heatmap", "correlation_heatmap.png"),
            ("Confusion Matrix", "confusion_matrix.png"),
            ("ROC Curve", "roc_curve.png"),
            ("Precision Recall Curve", "precision_recall_curve.png"),
            ("Feature Importance", "feature_importance.png"),
            ("Decision Tree", "decision_tree.png"),
            ("Training vs Testing", "train_test_accuracy.png"),
            ("Feature Distributions", "feature_distributions.png"),
            ("Missing Values", "missing_values.png")

        ]

        content = [

            section_title(
                "Model Analysis",
                "Decision Tree performance and visualizations"
            )

        ]

        for title, file_name in image_files:

            if os.path.exists(file_name):

                content.append(

                    ft.Container(
                        content=ft.Column([

                            ft.Text(
                                title,
                                color=green,
                                size=18,
                                weight=ft.FontWeight.BOLD
                            ),

                            ft.Image(
                                src=file_name,
                                width=750,
                                fit="contain"
                            )

                        ], spacing=10),

                        bgcolor=card_color,
                        padding=15,
                        border_radius=15
                    )

                )

        if len(content) == 1:

            content.append(
                ft.Text(
                    "No visualization images found. "
                    "Run ml_model.py first.",
                    color=muted
                )
            )

        return ft.Column(
            content,
            spacing=15,
            scroll=ft.ScrollMode.AUTO
        )

    # About page

    def about_page():

        return ft.Column([

            section_title(
                "About The Project",
                "Heart Disease Prediction using Machine Learning"
            ),

            ft.Container(
                content=ft.Column([

                    ft.Text(
                        "Decision Tree Classifier",
                        color=green,
                        size=23,
                        weight=ft.FontWeight.BOLD
                    ),

                    ft.Text(
                        "Decision Tree is a supervised machine learning "
                        "algorithm used for classification and regression.",
                        color=text_color,
                        size=15
                    ),

                    ft.Divider(color="#263746"),

                    ft.Text(
                        "How it works:",
                        color=cyan,
                        size=18
                    ),

                    ft.Text(
                        "1. The algorithm selects features to split data.\n"
                        "2. Each split creates smaller groups.\n"
                        "3. The process continues until a stopping "
                        "condition is reached.\n"
                        "4. The final leaf node provides the prediction.",
                        color=text_color
                    ),

                    ft.Text(
                        "Dataset: UCI Heart Disease (ID 45)",
                        color=cyan
                    ),

                    ft.Text(
                        "Target: 0 = No Disease, 1 = Disease",
                        color=cyan
                    ),

                    ft.Text(
                        "This project is designed for educational "
                        "and research purposes only.",
                        color=muted
                    )

                ], spacing=15),

                bgcolor=card_color,
                padding=25,
                border_radius=15
            )

        ], spacing=15)

    # Main content

    content_area = ft.Container(
        expand=True,
        padding=25
    )

    def navigate(name):

        pages = {
            "Dashboard": dashboard_page,
            "Prediction": prediction_page,
            "Model Analysis": analysis_page,
            "About": about_page
        }

        content_area.content = pages[name]()

        page.update()

    # Sidebar

    sidebar = ft.Container(
        width=230,
        bgcolor="#0D1721",
        padding=20,

        content=ft.Column([

            ft.Icon(
                ft.Icons.FAVORITE,
                color=green,
                size=40
            ),

            ft.Text(
                "MediCore AI",
                size=23,
                weight=ft.FontWeight.BOLD,
                color=green
            ),

            ft.Text(
                "HEART HEALTH SYSTEM",
                size=10,
                color=muted
            ),

            ft.Divider(color="#263746"),

            ft.Text(
                "MAIN MENU",
                color=muted,
                size=11
            ),

            ft.Button(
                content="Dashboard",
                icon=ft.Icons.DASHBOARD,
                bgcolor="#0D1721",
                color=text_color,
                on_click=lambda e: navigate("Dashboard")
            ),

            ft.Button(
                content="Prediction",
                icon=ft.Icons.FAVORITE,
                bgcolor="#0D1721",
                color=text_color,
                on_click=lambda e: navigate("Prediction")
            ),

            ft.Button(
                content="Model Analysis",
                icon=ft.Icons.ANALYTICS,
                bgcolor="#0D1721",
                color=text_color,
                on_click=lambda e: navigate("Model Analysis")
            ),

            ft.Button(
                content="About Project",
                icon=ft.Icons.INFO,
                bgcolor="#0D1721",
                color=text_color,
                on_click=lambda e: navigate("About")
            ),

            ft.Container(expand=True),

            ft.Text(
                "Decision Tree Model",
                color=green,
                size=12
            ),

            ft.Text(
                "Version 1.0",
                color=muted,
                size=11
            )

        ], spacing=12)
    )

    # App layout

    page.add(

        ft.Row([

            sidebar,

            ft.VerticalDivider(
                width=1,
                color="#263746"
            ),

            content_area

        ],
        expand=True,
        spacing=0)

    )

    navigate("Dashboard")


ft.run(main)