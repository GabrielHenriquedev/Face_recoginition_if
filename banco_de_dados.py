import sqlite3

# Conectando ao banco de dados (cria o arquivo se não existir)
conexao = sqlite3.connect("meu_banco.db")

# Criando uma tabela
cursor = conexao.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    matricula TEXT NOT NULL UNIQUE,
    foto_path TEXT NOT NULL
)
""")

# Salvando as alterações e fechando a conexão
conexao.commit()
conexao.close()

print("Banco de dados configurado com sucesso!")
