# Configuración de VS Code para compilar la memoria (LaTeX)

Este documento explica la configuración añadida para poder compilar la memoria
del TFG (`book.tex`) de forma cómoda desde VS Code, trabajando sobre WSL.

## 1. TeX Live instalado en WSL

Se instaló una distribución LaTeX en WSL (no la versión completa, sino un
conjunto reducido de paquetes suficiente para este proyecto):

```
sudo apt update && sudo apt install -y \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra \
    texlive-lang-spanish texlive-lang-european \
    texlive-bibtex-extra texlive-science texlive-pstricks \
    biber latexmk
```

Esto proporciona `pdflatex`, `latexmk` y `biber`, necesarios para compilar el
documento (usa glosarios, acrónimos, símbolos y bibliografía con biblatex).

## 2. Extensión de VS Code

Se instaló la extensión **LaTeX Workshop** (`James-Yu.latex-workshop`) en el
servidor remoto de WSL, mediante:

```
code --install-extension James-Yu.latex-workshop
```

Al abrir el workspace `/home/josue/TFG` desde VS Code conectado a WSL (Remote -
WSL), la extensión ya está disponible sin necesidad de instalarla manualmente
desde la interfaz.

## 3. Configuración en `.vscode/settings.json`

Se añadieron las siguientes claves al fichero `/home/josue/TFG/.vscode/settings.json`:

```json
"latex-workshop.latex.tools": [
    {
        "name": "latexmk-uah",
        "command": "latexmk",
        "args": [
            "-r",
            "%DIR%/../.latexmkrc",
            "-pdf",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-synctex=1",
            "%DOC%"
        ]
    }
],
"latex-workshop.latex.recipes": [
    {
        "name": "latexmk (memoria TFG)",
        "tools": ["latexmk-uah"]
    }
],
"latex-workshop.latex.recipe.default": "latexmk (memoria TFG)",
"latex-workshop.latex.rootFile.path": "${workspaceFolder}/memoria/TFG-LaTeX-Template-UAH/Book/book.tex",
"latex-workshop.latex.autoBuild.run": "onSave",
"latex-workshop.view.pdf.viewer": "tab"
```

### Qué hace cada bloque

- **`latex-workshop.latex.tools` / `.recipes`**: define una receta de
  compilación personalizada, equivalente al comando que se ejecutó
  manualmente para compilar la memoria por primera vez:

  ```
  latexmk -r ../.latexmkrc -pdf -interaction=nonstopmode -synctex=1 book.tex
  ```

  Se usa `-r %DIR%/../.latexmkrc` para que `latexmk` cargue el `.latexmkrc`
  de la plantilla (ubicado en `TFG-LaTeX-Template-UAH/`, un nivel por encima
  de `Book/`), que define las reglas necesarias para generar correctamente
  el glosario (`.glo` → `.gls`), los acrónimos (`.acn` → `.acr`) y los
  símbolos (`.sbl` → `.sym`) vía `makeindex`.

- **`latex-workshop.latex.rootFile.path`**: fija `Book/book.tex` como
  fichero raíz del proyecto. Así, aunque se edite y guarde un capítulo
  suelto (por ejemplo `chapters/introduccion.tex` o `appendix/manual.tex`),
  VS Code sabe que debe compilar siempre el documento completo (`book.tex`)
  y no el fichero individual.

- **`latex-workshop.latex.autoBuild.run: "onSave"`**: compila
  automáticamente cada vez que se guarda un `.tex` del proyecto, sin tener
  que lanzar el build a mano.

- **`latex-workshop.view.pdf.viewer: "tab"`**: abre el PDF resultante en una
  pestaña dentro de VS Code, con sincronización SyncTeX (clic en el PDF ↔
  clic en el código fuente).

## 4. Uso

1. Abrir el workspace `/home/josue/TFG` en VS Code conectado a WSL.
2. Editar cualquier fichero `.tex` de `memoria/TFG-LaTeX-Template-UAH/Book/`.
3. Guardar (`Ctrl+S`) — la memoria se recompila sola.
4. Para compilar manualmente: icono de "TeX" en la barra lateral, o
   `Ctrl+Alt+B`.

## 5. Limitación conocida

Al fijar `rootFile.path` a `book.tex`, esta configuración está pensada
específicamente para compilar la memoria. Si más adelante se quiere
compilar cómodamente desde VS Code otro documento del proyecto (por
ejemplo `anteproyecto.tex`, `slides.tex` o los PDFs de `PapeleoTFG/`), habrá
que ajustar o retirar temporalmente esta ruta fija.

## 6. Referencias sin resolver (pendiente)

En la última compilación quedaron 3 referencias sin resolver:

```
Reference `sub@fig:SRPvsModel_Fo1500_position1` on page 15 undefined
(líneas 314-316 del fichero correspondiente)
```

No impiden generar el PDF, pero están pendientes de revisión.
