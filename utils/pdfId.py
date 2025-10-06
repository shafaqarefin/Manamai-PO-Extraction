def create_pdf_id(data_dir):
    pdf_ids = []
    for file_path in data_dir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".pdf":
            continue

        pdf_id = file_path.name.split('_')[0]
        pdf_ids.append(pdf_id)
    return pdf_ids
