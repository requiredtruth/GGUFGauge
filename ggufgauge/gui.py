"""PySide6 dashboard for GGUF launch preflight."""
from __future__ import annotations
import argparse, json, struct, tempfile
from pathlib import Path
from .estimate import estimate_launch
from .gguf import read_model_info
from .system import detect_capacity


def _string(value: str) -> bytes:
    raw=value.encode(); return struct.pack("<Q",len(raw))+raw


def synthetic_gguf(path: Path, payload_bytes: int=2*1024**2) -> Path:
    fields=[("general.architecture",8,"llama"),("general.name",8,"Synthetic Llama"),("llama.block_count",4,32),("llama.embedding_length",4,4096),("llama.attention.head_count",4,32),("llama.attention.head_count_kv",4,8),("llama.context_length",4,8192)]
    parts=[b"GGUF",struct.pack("<IQQ",3,0,len(fields))]
    for key,kind,value in fields:
        parts += [_string(key),struct.pack("<I",kind),_string(str(value)) if kind==8 else struct.pack("<I",int(value))]
    header=b"".join(parts)
    with path.open("wb") as f: f.write(header); f.truncate(len(header)+payload_bytes)
    return path


def inspect(path: str, ram_gib: float|None=None, kv_type: str="f16") -> dict:
    model=read_model_info(path); capacity=detect_capacity(ram_bytes=int(ram_gib*1024**3) if ram_gib else None)
    estimate=estimate_launch(model,capacity,kv_type=kv_type)
    return {"model":model.to_dict(),"capacity":capacity.to_dict(),"estimate":estimate.to_dict()}


def run_demo() -> dict:
    with tempfile.TemporaryDirectory(prefix="ggufgauge-demo-") as tmp:
        path=synthetic_gguf(Path(tmp)/"synthetic.gguf")
        result=inspect(str(path),ram_gib=12.0,kv_type="q8_0")
        result["status"]="PASS"; result["demo"]="generated synthetic GGUF; no model download"; return result


def launch() -> int:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import QApplication,QComboBox,QDoubleSpinBox,QFileDialog,QFormLayout,QHBoxLayout,QLabel,QLineEdit,QMainWindow,QMessageBox,QPlainTextEdit,QProgressBar,QPushButton,QVBoxLayout,QWidget
    class Worker(QThread):
        done=Signal(object); failed=Signal(str)
        def __init__(self,fn): super().__init__(); self.fn=fn
        def run(self):
            try:self.done.emit(self.fn())
            except Exception as e:self.failed.emit(f"{type(e).__name__}: {e}")
    class Window(QMainWindow):
        def __init__(self):
            super().__init__();self.setWindowTitle("GGUFGauge Control Panel");self.resize(960,650);self.worker=None
            root=QWidget();layout=QVBoxLayout(root);layout.addWidget(QLabel("Container-aware GGUF memory and launch-envelope preflight"))
            form=QFormLayout();self.path=QLineEdit();browse=QPushButton("Browse…");row=QHBoxLayout();row.addWidget(self.path);row.addWidget(browse)
            self.ram=QDoubleSpinBox();self.ram.setRange(0,4096);self.ram.setSpecialValueText("Auto detect");self.ram.setSuffix(" GiB")
            self.kv=QComboBox();self.kv.addItems(["f16","q8_0","q5_1","q4_0"]);form.addRow("GGUF model",row);form.addRow("Available RAM",self.ram);form.addRow("KV type",self.kv);layout.addLayout(form)
            actions=QHBoxLayout();demo=QPushButton("Run Synthetic Demo");analyze=QPushButton("Analyze Selected Model");actions.addWidget(demo);actions.addWidget(analyze);layout.addLayout(actions)
            self.progress=QProgressBar();layout.addWidget(self.progress);self.status=QLabel("Ready");layout.addWidget(self.status);self.output=QPlainTextEdit();self.output.setReadOnly(True);layout.addWidget(self.output);self.setCentralWidget(root)
            browse.clicked.connect(self.pick);demo.clicked.connect(lambda:self.start("Generating safe GGUF demo…",run_demo));analyze.clicked.connect(self.analyze)
        def pick(self): p,_=QFileDialog.getOpenFileName(self,"Select GGUF","","GGUF (*.gguf);;All files (*)");self.path.setText(p)
        def analyze(self):
            p=self.path.text().strip()
            if not p:return QMessageBox.warning(self,"Model required","Select a GGUF file or run the synthetic demo.")
            ram=self.ram.value() or None;kv=self.kv.currentText();self.start("Reading bounded metadata…",lambda:inspect(p,ram,kv))
        def start(self,label,fn):
            if self.worker and self.worker.isRunning():return
            self.status.setText(label);self.progress.setRange(0,0);self.output.clear();self.worker=Worker(fn);self.worker.done.connect(self.finish);self.worker.failed.connect(self.fail);self.worker.start()
        def finish(self,v):self.progress.setRange(0,100);self.progress.setValue(100);self.status.setText("Complete");self.output.setPlainText(json.dumps(v,indent=2,sort_keys=True))
        def fail(self,e):self.progress.setRange(0,100);self.status.setText("Failed");self.output.setPlainText(e);QMessageBox.critical(self,"Preflight failed",e)
    app=QApplication([]);w=Window();w.show();return app.exec()


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--demo",action="store_true");a=p.parse_args()
    if a.demo:print(json.dumps(run_demo(),indent=2,sort_keys=True));return 0
    return launch()
if __name__=="__main__":raise SystemExit(main())
