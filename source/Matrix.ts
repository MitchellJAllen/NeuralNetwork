class Matrix {
	private rowCount: number;
	private columnCount: number;
	private entries: number[];

	public constructor(rowCount: number, columnCount: number) {
		this.rowCount = rowCount | 0;
		this.columnCount = columnCount | 0; // remove fractional parts //

		if (this.rowCount < 0) {
			this.rowCount = 0;
		}

		if (this.columnCount < 0) {
			this.columnCount = 0;
		}

		let entryCount = this.rowCount * this.columnCount;

		this.entries = new Array<number>(entryCount);
	}

	private getIndex(rowIndex: number, columnIndex: number): number {
		let row = rowIndex | 0;
		let column = columnIndex | 0; // remove fractional parts //

		if (row < 0 || row >= this.rowCount || column < 0 || column >= this.columnCount) {
			console.error("Matrix index (" + row + "," + column + ") out of bounds");

			return -1;
		}

		return row * this.columnCount + column;
	}

	public getValue(rowIndex: number, columnIndex: number): number {
		let index = this.getIndex(rowIndex, columnIndex);

		if (index == -1) {
			return undefined;
		}

		return this.entries[index];
	}

	public setValue(rowIndex: number, columnIndex: number, value: number): void {
		let index = this.getIndex(rowIndex, columnIndex);

		if (index == -1) {
			return;
		}

		this.entries[index] = value;
	}

	public getRowCount(): number {
		return this.rowCount;
	}

	public getColumnCount(): number {
		return this.columnCount;
	}

	public getEntryCount(): number {
		return this.entries.length;
	}
}
