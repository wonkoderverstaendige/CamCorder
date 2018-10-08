"""Qt5 demo GUI cobbled together from StackOverflow answers."""
import sys
import cv2

from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, parent):
        super().__init__(parent)

    def run(self):
        print('starting thread')
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()
        self.show()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(self.width, self.height)

        # create a label
        self.label = QLabel(self)
        self.label.move(0, 0)
        self.label.resize(640, 480)

        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in [Qt.Key_Escape, Qt.Key_Q]:
            self.close()
            # Don't let the app hang!
            QApplication.quit()

        if event.key() == Qt.Key_Space:
            print('Pause')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())