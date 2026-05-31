# GDP Particle Filter Streamlit App

This app runs an interactive particle filter model for GDP output gap estimation.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Uploading another country's GDP CSV

Before uploading your own CSV, rename the required columns exactly:

- `date`: the observation date
- `GDP`: the real GDP level

Example:

| date | GDP |
|---|---:|
| 1990-01-01 | 1234.5 |
| 1990-04-01 | 1251.8 |

The app uses these columns to calculate GDP growth, fit a polynomial trend, build the proxy output gap, and run the filters.

## Polynomial degree

The app lets the user select the polynomial degree. For other countries, choose the degree that gives the best-looking smooth long-run GDP trend. Higher is not always better. If the trend bends wildly at the beginning/end or chases short-term movements, lower the degree.
