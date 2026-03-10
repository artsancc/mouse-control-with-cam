# Kamera ile Fare Kontrolü

Farneback Dense Optik Akış kullanarak webcam görüntüsündeki el hareketlerini algılar ve fare imlecini kontrol eder. Hızlı yatay sallama hareketi ile sol tıklama yapılabilir.


## Özellikler

- Gerçek zamanlı webcam görüntüsü işleme
- El hareketi ile fare imleci kontrolü
- Hızlı yatay sallama → Sol tıklama
- FPS sayacı ve anlık hareket bilgisi (HUD)
- Farneback akış vektörü ve HSV renk haritası görselleştirmesi
- Kolayca ayarlanabilir hassasiyet ve eşik parametreleri

---

## Gereksinimler

| Paket |  Açıklama |
|-------|----------|
| Python | Ana dil |
| opencv-python | Görüntü işleme ve optik akış |
| numpy | Matris işlemleri |
| pyautogui | Fare ve klavye kontrolü |

---

## Gereksinimleri Yükleyin
```bash
pip install opencv-python numpy pyautogui
```
---

## Kontroller

| Hareket / Tuş | Eylem |
|---------------|-------|
| El hareketi | Fare imlecini hareket ettir |
| Hızlı sağ-sol sallama | Sol tıklama |
| `q` veya `ESC` | Uygulamadan çık |


