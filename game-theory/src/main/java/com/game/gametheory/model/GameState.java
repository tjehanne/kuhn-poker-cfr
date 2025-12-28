package com.game.gametheory.model;

public class GameState {

    private int day;
    private int hawks;
    private int doves;

    public GameState(int day, int hawks, int doves) {
        this.day = day;
        this.hawks = hawks;
        this.doves = doves;
    }

    public int getDay() {
        return day;
    }

    public int getHawks() {
        return hawks;
    }

    public int getDoves() {
        return doves;
    }
}
