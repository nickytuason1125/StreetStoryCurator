import clsx from 'clsx';
import type { ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Merge class names, letting a later utility win over an earlier conflicting
 *  one. Both deps were already in package.json but unused — `className` never
 *  appeared once in the old 4,868-line App.tsx. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
