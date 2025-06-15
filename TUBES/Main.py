import cv2
import numpy as np
import utlis
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
from datetime import datetime

# Inisialisasi dan Konfigurasi
tinggiGambar = 700
lebarGambar = 700
jumlahSoal = 5
jumlahPilihan = 5
jawabanBenar = [1, 2, 0, 2, 4]

# Menu input
print("Pilih mode input:")
print("1. Webcam (kamera laptop)")
print("2. Upload gambar dari file")
mode = input("Masukkan pilihan (1/2): ")

# Inisialisasi input
if mode == "1":
    pakaiWebcam = True
    cap = cv2.VideoCapture(1)  # webcam internal
    cap.set(10, 160)
else:
    pakaiWebcam = False
    root = tk.Tk()
    root.withdraw()
    pathImage = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if pathImage == "":
        print("Tidak ada gambar dipilih. Program dihentikan.")
        exit()

hitung = 0

while True:
    if pakaiWebcam:
        sukses, img = cap.read()
    else:
        img = cv2.imread(pathImage)

    img = cv2.resize(img, (lebarGambar, tinggiGambar))
    imgAsli = img.copy()
    imgKosong = np.zeros((tinggiGambar, lebarGambar, 3), np.uint8)

    # Proses awal
    imgAbu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgAbu, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 70)

    try:
        imgKontur = img.copy()
        imgKonturBesar = img.copy()
        kontur, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgKontur, kontur, -1, (0, 255, 0), 10)

        konturPersegi = utlis.rectContour(kontur)
        titikLembar = utlis.getCornerPoints(konturPersegi[0])
        titikNilai = utlis.getCornerPoints(konturPersegi[1])

        if titikLembar.size != 0 and titikNilai.size != 0:
            titikLembar = utlis.reorder(titikLembar)
            pts1 = np.float32(titikLembar)
            pts2 = np.float32([[0, 0], [lebarGambar, 0], [0, tinggiGambar], [lebarGambar, tinggiGambar]])
            matriks = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matriks, (lebarGambar, tinggiGambar))

            titikNilai = utlis.reorder(titikNilai)
            ptsG1 = np.float32(titikNilai)
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matriksNilai = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgNilai = cv2.warpPerspective(img, matriksNilai, (325, 150))

            imgWarpAbu = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
            imgThreshold = cv2.threshold(imgWarpAbu, 170, 255, cv2.THRESH_BINARY_INV)[1]

            kotak = utlis.splitBoxes(imgThreshold)

            nilaiPixel = np.zeros((jumlahSoal, jumlahPilihan))
            countR, countC = 0, 0
            for image in kotak:
                total = cv2.countNonZero(image)
                nilaiPixel[countR][countC] = total
                countC += 1
                if countC == jumlahPilihan:
                    countC = 0
                    countR += 1

            jawabanUser = []
            for x in range(0, jumlahSoal):
                arr = nilaiPixel[x]
                index = np.where(arr == np.amax(arr))
                jawabanUser.append(index[0][0])

            hasilNilai = []
            for x in range(0, jumlahSoal):
                if jawabanUser[x] == jawabanBenar[x]:
                    hasilNilai.append(1)
                else:
                    hasilNilai.append(0)

            skor = (sum(hasilNilai) / jumlahSoal) * 100

            # === Export ke Excel ===
            os.makedirs("Scan", exist_ok=True)
            fileRekap = "Scan/rekap_omr.xlsx"

            barisBaru = {
                "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Skor": f"{skor:.2f}%",
                **{f"Q{i+1}": jawabanUser[i] for i in range(jumlahSoal)}
            }

            if os.path.exists(fileRekap):
                df_lama = pd.read_excel(fileRekap)
                df_baru = pd.concat([df_lama, pd.DataFrame([barisBaru])], ignore_index=True)
            else:
                df_baru = pd.DataFrame([barisBaru])

            df_baru.to_excel(fileRekap, index=False)
            print(f"Data ditambahkan ke: {fileRekap}")

            # Tampilkan hasil visual
            utlis.showAnswers(imgWarp, jawabanUser, hasilNilai, jawabanBenar)
            utlis.drawGrid(imgWarp)

            imgKosongDraw = np.zeros_like(imgWarp)
            utlis.showAnswers(imgKosongDraw, jawabanUser, hasilNilai, jawabanBenar)
            matriksBalik = cv2.getPerspectiveTransform(pts2, pts1)
            imgHasil = cv2.warpPerspective(imgKosongDraw, matriksBalik, (lebarGambar, tinggiGambar))

            imgKosongNilai = np.zeros_like(imgNilai)
            cv2.putText(imgKosongNilai, str(int(skor)) + "%", (70, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
            matriksBalikNilai = cv2.getPerspectiveTransform(ptsG2, ptsG1)
            imgHasilNilai = cv2.warpPerspective(imgKosongNilai, matriksBalikNilai, (lebarGambar, tinggiGambar))

            imgGabung = cv2.addWeighted(imgAsli, 1, imgHasil, 1, 0)
            imgGabung = cv2.addWeighted(imgGabung, 1, imgHasilNilai, 1, 0)
            imgAsli = imgGabung

            cv2.imshow("Hasil Akhir", imgGabung)

        else:
            raise Exception("Gagal menemukan kontur utama")

    except Exception as e:
        print("Terjadi error saat pemrosesan:", str(e))
        imgThreshold = np.zeros((tinggiGambar, lebarGambar), np.uint8)
        imgWarp = np.zeros((tinggiGambar, lebarGambar, 3), np.uint8)
        imgKonturBesar = np.zeros((tinggiGambar, lebarGambar, 3), np.uint8)
        imgAsli = np.zeros((tinggiGambar, lebarGambar, 3), np.uint8)

    gambarArray = ([img, imgAbu, imgCanny, imgKontur], [imgKonturBesar, imgThreshold, imgWarp, imgAsli])
    labelArray = [["Original","Gray","Edges","Contours"],
                  ["Biggest Contour","Threshold","Warped","Final"]]
    imgStack = utlis.stackImages(gambarArray, 0.5, labelArray)
    cv2.imshow("Proses", imgStack)

    # Simpan hasil kalau tekan tombol 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scan/hasil_" + str(hitung) + ".jpg", imgAsli)
        print("Gambar disimpan!")
        hitung += 1

    # Khusus untuk mode upload, selesai sekali -> keluar loop
    if not pakaiWebcam:
        cv2.waitKey(0)
        break
