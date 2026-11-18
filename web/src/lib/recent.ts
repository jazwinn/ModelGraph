// Recently opened/saved graph files, kept in this browser.

export interface RecentFile {
  file: string; // name in graphs/ or absolute path
  name: string; // model name at the time
  at: number;
}

const KEY = 'mg-recent';
const LIMIT = 10;

export function getRecent(): RecentFile[] {
  try {
    const list = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

export function addRecent(file: string, name: string) {
  if (!file) return;
  const list = [{ file, name, at: Date.now() }, ...getRecent().filter((r) => r.file !== file)].slice(0, LIMIT);
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* storage unavailable */
  }
}

export function removeRecent(file: string) {
  try {
    localStorage.setItem(KEY, JSON.stringify(getRecent().filter((r) => r.file !== file)));
  } catch {
    /* storage unavailable */
  }
}
