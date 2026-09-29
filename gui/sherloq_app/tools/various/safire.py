from gui.sherloq_app.ui.localization import choice
import colorsys
import cv2 as cv
import numpy as np
from PySide6.QtWidgets import QComboBox,QSpinBox,QDoubleSpinBox,QHBoxLayout,QLabel
from ...ui.research_panel import ResearchPanel,display as map_display

def colors(n):return np.asarray([[round(c*255) for c in colorsys.hsv_to_rgb((.76+i*.61803398875)%1,.75,.95)[::-1]] for i in range(n)],np.uint8)
def display(request):
    image,result,mode=request
    if result['metadata']['binary']:return map_display((image,result,min(mode,1)))
    if mode==2:return map_display((image,result,1))
    palette=colors(result['metadata']['sources']);labels=result['source_labels'];heat=cv.resize(palette[labels],(image.shape[1],image.shape[0]),interpolation=cv.INTER_NEAREST)
    return cv.addWeighted(image,.5,heat,.5,0) if mode==0 else heat

class SafireWidget(ResearchPanel):
    def __init__(self,image,parent=None):
        super().__init__(image,'safire',['Multisource — k-means','Multisource — DBSCAN','Binaire'],
            'Séparation de sources probables à 1024 × 1024 px. Les couleurs identifient des groupes, sans désigner à elles seules une région falsifiée.',parent)
        self.side=QComboBox();self.side.addItems(['16','32']);self.groups=QSpinBox();self.groups.setRange(1,16);self.groups.setValue(3);self.eps=QDoubleSpinBox();self.eps.setDecimals(3);self.eps.setRange(.001,100);self.eps.setSingleStep(.05);self.eps.setValue(.2);self.minimum=QSpinBox();self.minimum.setRange(1,1024);self.minimum.setValue(1)
        row=QHBoxLayout()
        for label,control in [('Points / côté',self.side),('Sources',self.groups),('Distance DBSCAN',self.eps),('Support DBSCAN',self.minimum)]:row.addWidget(QLabel(label));row.addWidget(control)
        row.addStretch();self.layout().insertLayout(1,row);self.mode.addItem('Confiance');self.mode.setItemText(1,'Sources');self.draw_job.compute=display
        for c in (self.side,self.groups,self.eps,self.minimum):(c.currentIndexChanged if isinstance(c,QComboBox) else c.valueChanged).connect(self.changed)
        self.variant.currentIndexChanged.connect(self.options);self.options()
    def options(self):
        if not hasattr(self,'side'):return
        db='DBSCAN' in choice(self.variant);binary=choice(self.variant)=='Binaire';busy=self.job.is_busy
        self.groups.setEnabled(not busy and not db and not binary);self.eps.setEnabled(not busy and db);self.minimum.setEnabled(not busy and db);self.side.setEnabled(not busy)
        self.mode.setItemText(1,'Carte' if binary else 'Sources')
        self.legend.setText('Deux groupes forcés ; le plus petit est affiché comme candidat.' if binary else 'Les couleurs représentent les sources regroupées.')
    def params(self):
        result=super().params()
        if hasattr(self,'side'):result.update(side=int(choice(self.side)),groups=self.groups.value(),eps=self.eps.value(),minimum=self.minimum.value())
        return result
    def state(self,busy):super().state(busy);self.options()
    def complete(self,result):
        super().complete(result)
        if not result['metadata']['binary']:
            labels=result['source_labels'];items=[]
            for i,color in enumerate(colors(result['metadata']['sources'])):
                b,g,r=map(int,color);items.append(f'<span style="color:rgb({r},{g},{b})">■</span> Source {i+1} : {np.mean(labels==i)*100:.1f} %')
            self.legend.setText(' · '.join(items))
