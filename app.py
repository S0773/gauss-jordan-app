import streamlit as st
import numpy as np

# CONFIG
st.set_page_config(page_title="Gauss-Jordan Solver", layout="wide")

st.title("Solucionador Ax = b")

# INPUTS
st.sidebar.header("Matriz A (3x3)")
A = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        A[i,j] = st.sidebar.number_input(f"A{i+1}{j+1}", value=0.0)

st.sidebar.header("Vector b")
b = np.zeros(3)

for i in range(3):
    b[i] = st.sidebar.number_input(f"b{i+1}", value=0.0)

# FUNCIÓN GAUSS-JORDAN
def gauss_jordan(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    for i in range(n):
        # Pivot
        max_row = np.argmax(abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Normalizar
        pivot = A[i, i]
        if pivot == 0:
            raise ValueError("Sistema sin solución única")

        A[i] = A[i] / pivot
        b[i] = b[i] / pivot

        # Eliminar
        for j in range(n):
            if j != i:
                factor = A[j, i]
                A[j] = A[j] - factor * A[i]
                b[j] = b[j] - factor * b[i]

    return b

# BOTÓN
if st.button("Resolver"):
    try:
        x = gauss_jordan(A.copy(), b.copy())

        st.subheader("Solución")
        st.write(f"x1 = {x[0]:.4f}")
        st.write(f"x2 = {x[1]:.4f}")
        st.write(f"x3 = {x[2]:.4f}")

        # Verificación
        error = np.linalg.norm(A @ x - b)
        st.write(f"Error: {error:.6f}")

    except Exception as e:
        st.error(str(e))

