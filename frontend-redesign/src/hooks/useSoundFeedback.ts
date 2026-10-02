import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "visionqc_sound_enabled";

export function useSoundFeedback() {
  const [enabled, setEnabled] = useState<boolean>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved !== null ? saved === "true" : true;
    } catch {
      return true;
    }
  });

  const toggleSound = useCallback(() => {
    setEnabled((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(STORAGE_KEY, String(next));
      } catch {}
      return next;
    });
  }, []);

  const playTone = useCallback(
    (freqs: number[], type: OscillatorType = "sine", duration = 0.12) => {
      if (!enabled || typeof window === "undefined") return;
      try {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (!AudioCtx) return;
        const ctx = new AudioCtx();
        let startTime = ctx.currentTime;

        freqs.forEach((freq, idx) => {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();

          osc.type = type;
          osc.frequency.setValueAtTime(freq, startTime + idx * (duration * 0.8));

          gain.gain.setValueAtTime(0.15, startTime + idx * (duration * 0.8));
          gain.gain.exponentialRampToValueAtTime(0.001, startTime + (idx + 1) * duration);

          osc.connect(gain);
          gain.connect(ctx.destination);

          osc.start(startTime + idx * (duration * 0.8));
          osc.stop(startTime + (idx + 1) * duration);
        });

        setTimeout(() => {
          ctx.close().catch(() => {});
        }, (freqs.length + 1) * duration * 1000);
      } catch (e) {
        // AudioContext might be blocked until user gesture, ignore
      }
    },
    [enabled]
  );

  const playPass = useCallback(() => {
    playTone([523.25, 659.25, 783.99], "sine", 0.1); // C5, E5, G5 happy chord
  }, [playTone]);

  const playFail = useCallback(() => {
    playTone([329.63, 246.94], "triangle", 0.15); // E4, B3 warning
  }, [playTone]);

  const playReview = useCallback(() => {
    playTone([440, 493.88], "sine", 0.12); // A4, B4
  }, [playTone]);

  const playClick = useCallback(() => {
    playTone([800], "sine", 0.04);
  }, [playTone]);

  return {
    enabled,
    toggleSound,
    playPass,
    playFail,
    playReview,
    playClick,
  };
}
