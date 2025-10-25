from datetime import datetime


def formatDate(date: str):
    date_str = date
    formatted = datetime.strptime(date_str, "%d %b, %Y").strftime("%Y-%m-%d")
    return formatted
