from ...ui.research_panel import ResearchPanel

class AdaptiveCFAWidget(ResearchPanel):
    def __init__(self,image,parent=None):
        super().__init__(image,'adaptive_cfa',['Original','JPEG 95','Sans JPEG'],
            'Cohérence locale de mosaïque CFA ; les traitements et rééchantillonnages peuvent effacer ces traces.',parent)
        # Small probability rounding differences can flip the published hard
        # grid decision. Keep CPU as default until GPU decisions are equivalent.
        self.cpu.setChecked(True)
