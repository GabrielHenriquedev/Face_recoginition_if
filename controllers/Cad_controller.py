from PyQt5.QtWidgets import QFileDialog


def browse_photo(parent, photo_input):
    """
    Abre o diálogo de seleção de arquivos para escolher uma imagem e
    atualiza o campo de entrada de texto com o caminho da imagem selecionada.
    """
    file_name, _ = QFileDialog.getOpenFileName(parent, "Selecionar Foto", "", "Imagens (*.png *.jpg *.jpeg)")

    # Se um arquivo foi selecionado, o caminho é exibido no campo de texto fornecido
    if file_name:
        photo_input.setText(file_name)
