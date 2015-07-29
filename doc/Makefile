all: ZacharyBookey.pdf

ZacharyBookey.pdf: ZacharyBookey.tex ZacharyBookey.bib
	pdflatex ZacharyBookey
	bibtex ZacharyBookey
	pdflatex ZacharyBookey
	pdflatex ZacharyBookey

clean:
	rm ZacharyBookey.pdf ZacharyBookey.bbl ZacharyBookey.blg ZacharyBookey.log ZacharyBookey.aux
