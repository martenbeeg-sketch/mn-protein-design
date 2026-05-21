/**
 * Convert a number into a Float64Array
 * @param num number
 * @returns float64array
 */
const toBytesFloat64 = (num: number) => {
    const arr = new ArrayBuffer(8); // 8 bytes
    const view = new DataView(arr);
    view.setFloat64(0, num, true); // byteOffset = 0; litteEndian = false
    return arr;
};

export { toBytesFloat64 };