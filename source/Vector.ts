class Vector {
	private entries: number[];

	public constructor(entryCount: number) {
		let entryCountInteger = entryCount | 0; // remove fractional part //

		if (entryCountInteger < 0) {
			entryCountInteger = 0;
		}

		this.entries = new Array<number>(entryCountInteger);
	}

	private getIndex(entryIndex: number): number {
		let entry = entryIndex | 0; // remove fractional part //

		if (entry < 0 || entry >= this.entries.length) {
			console.error("Vector index (" + entry + ") out of bounds");

			return -1;
		}

		return entry;
	}

	public getValue(entryIndex: number): number {
		let index = this.getIndex(entryIndex);

		if (index == -1) {
			return undefined;
		}

		return this.entries[index];
	}

	public setValue(entryIndex: number, value: number): void {
		let index = this.getIndex(entryIndex);

		if (index == -1) {
			return;
		}

		this.entries[index] = value;
	}

	public getEntryCount(): number {
		return this.entries.length;
	}

	public getEntryValues(): number[] {
		return this.entries.slice(); // creates a copy of the entries array //
	}
}
